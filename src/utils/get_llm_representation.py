import re, os
import argparse
from typing import List

# Third-party imports
import pandas as pd
import sys

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
    parser = argparse.ArgumentParser(description="Enhance topic representation using VLLM")
    
    # File paths
    parser.add_argument("--few_shot_prompt_txt_path",
                       default="/home/guest/r12922050/GitHub/d2qplus/prompts/enhance_NL_topic.txt", 
                       help="Path to few-shot prompt text file")
    parser.add_argument("--topic_base_dir", type=str, required=True, help="Base directory for topic model files")
    
    # Model parameters
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name for VLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    topic_model_df_pickle_path = f"{args.topic_base_dir}/topic_info_dataframe.pkl"
    messages = construct_prompt_messages(topic_model_df_pickle_path, args.few_shot_prompt_txt_path)

    # vllm part
    llm = LLM(model=args.model, 
              tensor_parallel_size=args.tensor_parallel_size, 
              max_model_len=args.max_model_len, 
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    generated_topics = enhance_topic_representation_with_vllm(messages=messages, llm=llm, sampling_params=sampling_params)

    df = pd.read_pickle(topic_model_df_pickle_path)
    df['Enhanced_Topic'] = generated_topics

    enhanced_topic_df = df
    enhanced_topic_df.to_csv(f"{args.topic_base_dir}/topic_info_dataframe_enhanced.csv", index=False)
    print(f"Enhanced topic model saved to {args.topic_base_dir}/topic_info_dataframe_enhanced.csv")
    
    enhanced_topic_df.to_pickle(f"{args.topic_base_dir}/topic_info_dataframe_enhanced.pkl")
    print(f"Enhanced topic model saved to {args.topic_base_dir}/topic_info_dataframe_enhanced.pkl")
    
if __name__ == "__main__":
    main()
