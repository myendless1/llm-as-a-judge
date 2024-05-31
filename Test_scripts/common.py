from typing import Optional
import dataclasses
from enum import auto, Enum
import json
from typing import List, Tuple, Any

import torch
from transformers import StoppingCriteria

def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            # print("review: ", review)
            # raise Exception('Invalid score pair.')
            raise Exception()
            pass
    except Exception as e:
        # print(f'{e}\nContent: {review}\n'
        #              'You must manually fix the score pair.')
        return [-1, -1]


def translate_score_to_win_list(score_list, T=0.0):
    win_list = []
    for i in range(len(score_list)):
        if score_list[i][0] - score_list[i][1] > T:
            win_list.append(1)
        elif score_list[i][1] - score_list[i][0] > T:
            win_list.append(-1)
        else:
            win_list.append(0)
    return win_list

def generate_question_template(domain, question1, question2):
    Q = ("Human", "Provide a question in [" + domain + "] domain just like {" + question1 + "}, your provided question must be different from the questions that we have mentioned in this conversation.")
    A = ("Assistant", "Certainly! Here's another question in a [" + domain + "] domain: {" + question2 + "}")
    return (Q, A)

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

# create num to words dict
num2words = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five",
             6:"six", 7:"seven", 8: "eight", 9: 'nine', 10: 'ten', \
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    prompt: str
    template: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    appendix: str = "### Response: "

    skip_next: bool = False
    conv_id: Any = None
    answer_format: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self, answer_num=None):
        if answer_num is not None:
            prompt = self.prompt\
                .replace("two", num2words[int(answer_num)])\
                .replace("for Assistant 1 and 2", "for Assistant 1")
            
            plug_in_after_str = "for Assistant 1"
            plug_in_pos = prompt.find(plug_in_after_str) + len(plug_in_after_str)

            new_answer = ""
            for i in range(int(answer_num)-2):
                new_answer += f", {i+2}"
            new_answer += f" and {int(answer_num)}"
            prompt = prompt[:plug_in_pos] + new_answer + prompt[plug_in_pos:]
        else:
            prompt = self.prompt
        return Conversation(
            system=self.system,
            roles=self.roles,
            prompt=prompt,
            prompt_template=self.prompt_template,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            appendix=self.appendix,
            conv_id=self.conv_id,
            answer_format=self.answer_format)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "prompt": self.prompt,
            "prompt_template": self.prompt_template,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "appendix": self.appendix,
            "conv_id": self.conv_id,
        }

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


conv_judge_chatglm = Conversation(
    system='''<|im_start|>system
You are a helpful assistant in evaluating the quality of the responses of two assistants for a given instruction. Your goal is to select the better response for the given instruction. The two responses are generated by two different AI assistants respectively.
<|im_end|>''',
    prompt="",
    template='''<|im_start|>user
After giving a brief explanation, select the A or B that is better for the given instruction. Output your decision and explanation in the Decision & Explanation section.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Assistant A's response and Assistant B's response are **equally likely** to be the better.

You should first provide a brief explanation of your evaluation, and then always end your response with "Final Verdict: [[A]]" or "Final Verdict: [[B]]" or "Final Verdict: [[C]] for tie".
[[A]] means the Assistant A's response is better, [[B]] means the Assistant B's response is better, [[Tie]] means the responses are both ok or none of them is suitable.
Do NOT output any other words.
Do NOT say your verdict at the beginning. You better do reasoning and thinking **before** claiming which is better.

# Instruction:
{query}

# Assistant A's response:
{response1}

# Assistant B's response:
{response2}

# Decision & Explanation:
<|im_end|>
''',
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="\n<|im_start|>assistant\n"
)

conv_judge_pair_gemma = Conversation(
    system='''<start_of_turn>system
You are a helpful assistant in evaluating the quality of the responses of two assistants for a given instruction. Your goal is to select the better response for the given instruction. The two responses are generated by two different AI assistants respectively.
<end_of_turn>''',
    prompt="",
    template='''<start_of_turn>user
After giving a brief explanation, select the A or B that is better for the given instruction. Output your decision and explanation in the Decision & Explanation section.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Assistant A's response and Assistant B's response are **equally likely** to be the better.

You should first provide a brief explanation of your evaluation, and then always end your response with "Final Verdict: [[A]]" or "Final Verdict: [[B]]" or "Final Verdict: [[C]] for tie"
[[A]] means the Assistant A's response is better, [[B]] means the Assistant B's response is better, [[Tie]] means the responses are both ok or none of them is suitable.
Do NOT output any other words.
Do NOT say your verdict at the beginning. You better do reasoning and thinking **before** claiming which is better.

# Instruction:
{query}

# Assistant A's response:
{response1}

# Assistant B's response:
{response2}

# Decision & Explanation:
<end_of_turn>
''',
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="\n<start_of_turn>model\n"
)