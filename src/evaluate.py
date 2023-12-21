from configs import get_configs
from tqdm import tqdm
import ast
import re
from fastchat.model.model_adapter import get_conversation_template
from fastchat.llm_judge.common import chat_compeletion_openai
from fastchat.llm_judge.common import load_judge_prompts

import openai

# ATTENTION: This API key is purely used for testing your evaluation code. 
# The expense is limited, do not abuse it. You should change the api to your 
# own when performing large-scale evaluation
openai.api_base = "https://api.lmtchina.com/v1"
openai.api_key = "sk-COPn4H3E4YTtWv95F5Eb82EcE9034aFfBd0f8f7672879fE6"

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

class ChatGPTEvaluator:
    """
    Usage:
    >>> evaluator = ChatGPTEvaluator()
    >>> questions = ["what's the tallest building in the world?", "1+1=?"]
    >>> answers = ["The world's tallest human-made structure is the 828-metre-tall (2,717 ft) Burj Khalifa in Dubai, United Arab Emirates.", "3"]
    >>> judges = evaluator = evaluator.get_judges(questions, answers)
    >>> ratings, prompts, responses = zip(*judges)
    """
    def __init__(self) -> None:
        self.judge_prompts = self.prepare_judge()
        # self.response_model = "gpt-4"
        self.response_model = "gpt-3.5-turbo" # see https://openai.com/pricing for available models
    
    def prepare_judge(self, judge_file=None):
        if judge_file is None:
            judge_file = "judge_prompts.jsonl"
        return load_judge_prompts(judge_file)["single-v1"]

    def _get_rating(self, question, answer, temperature=0, max_tokens=2048):
        """Given the question and answer pair, get the rating from the model
        Return:
            rating (int): the score of the answer, from 0 to 10.
            user_prompt (str): the input text to ChatGPT.
            response (str): the output text of ChatGPT, which contains the justification of the score.
        """
        user_prompt = self.judge_prompts["prompt_template"].format(
            question=question,
            answer=answer,
        )
        sys_prompt = self.judge_prompts["system_prompt"]
        conv = get_conversation_template(self.response_model)
        conv.set_system_message(sys_prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        
        # get the judment response from the GPT3.5/GPT4 model
        response = chat_compeletion_openai(self.response_model, conv, temperature=temperature, max_tokens=max_tokens)
        match = re.search(one_score_pattern, response)
        if not match:
            match = re.search(one_score_pattern_backup, response)
        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
            print("cannot find rating from GPT4!")
        return rating, user_prompt, response

    def get_judges(self, questions, answers, parallel=16):
        """evaluate the quality of answers w.r.t. to questions. The input should be text string, NOT token
        Args:
            questions (list): [question1, question2, ...]
            anwsers (list): [answer1, answer2, ...]
            parallel (Optional int): evaluate how many multiple (question, answer) pairs simultaneously.
        Return:
            judges (list): [judge1, judge2, ...], where each judge is a tuple of (rating, user_prompt, response)
        """
        from concurrent.futures import ThreadPoolExecutor
        judges = []
        parallel = min(parallel, len(questions))
        with ThreadPoolExecutor(parallel) as executor:
            for judge in tqdm(executor.map(self._get_rating, questions, answers)):
                judges.append(judge)
        return judges