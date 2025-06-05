import re, os
from typing import List

# Third-party imports
import pandas as pd
from vllm import LLM, SamplingParams
from util import read_txt
# Local imports
from constants import TOPIC_REPRESENTATION_SYSTEM_PROMPT


def extract_topic_label(text: str) -> str:
    """Extract text after 'topic:' label."""
    pattern = r"topic:\s*(.+)$"
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # group(1) gets text inside brackets
    return text

def construct_prompt_messages(topic_model_df_pickle_path: str, few_shot_prompt_txt_path: str):
    # construct messages list
    df = pd.read_pickle(topic_model_df_pickle_path) #Topic, Count, Name, Representation (list of keywords), Representative_Docs (list of sentences)
    few_shot_prompt = read_txt(few_shot_prompt_txt_path)
    messages = []
    for index, row in df.iterrows():
        keywords = ", ".join(row['Representation'])
        sentences = "\n".join([f"- {sent}" for sent in row['Representative_Docs']])
        prompt = few_shot_prompt.replace("[DOCUMENTS]", sentences).replace("[KEYWORDS]", keywords)
        messages.append(
            [
                {
                    "role": "system", 
                    "content": TOPIC_REPRESENTATION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    return messages
    
def enhance_topic_representation_with_vllm(
        messages: List[List],
        llm: LLM,
        sampling_params: SamplingParams
    ) -> List[str]:

    generated_topics = []
    outputs = llm.chat(messages, sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        topic_label = extract_topic_label(generated_text)
        generated_topics.append(topic_label)
    return generated_topics



def main():
    topic_model_df_pickle_path = "/home/guest/r12922050/GitHub/d2qplus/topics/nfcorpus/topic_model.pickle"
    few_shot_prompt_txt_path = "/home/guest/r12922050/GitHub/d2qplus/prompts/enhance_NL_topic.txt"
    output_enhanced_topic_model_df_pickle_path = "/home/guest/r12922050/GitHub/d2qplus/topics/nfcorpus/topic_model_enhanced.pickle"
    messages = construct_prompt_messages(topic_model_df_pickle_path, few_shot_prompt_txt_path)

    # vllm part
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2, max_model_len=8192, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=128,
        # guided_decoding=GuidedDecodingParams(regex=r"topic:\s*<[^>]+>")
    )

    generated_topics = enhance_topic_representation_with_vllm(messages=messages, llm=llm, sampling_params=sampling_params)

    df = pd.read_pickle(topic_model_df_pickle_path)
    df['Enhanced_Topic'] = generated_topics

    enhanced_topic_df = df
    enhanced_topic_df.to_pickle(output_enhanced_topic_model_df_pickle_path)
    print(f"Enhanced topic model saved to {output_enhanced_topic_model_df_pickle_path}")
if __name__ == "__main__":
    main()
