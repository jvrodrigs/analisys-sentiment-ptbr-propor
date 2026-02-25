import os, time
import pandas as pd
import common.extractly as ExtractlyCommon

from openai import OpenAI
from datetime import timedelta

path_run_code = ""
chatgpt_key = ""
MAX_RETRIES = 3

llm = (
    "gpt-5",
    "gpt-5-nano",
    "gpt-oss-120b-MXFP4",    	
    "gpt-oss-20b-MXFP4",               		
    
    "DeepSeek-R1-Distill-Llama-70B",  		
    "DeepSeek-R1-Distill-Llama-8B",   		
    "DeepSeek-R1-Distill-Qwen-14B",   		
    "DeepSeek-R1-Distill-Qwen-1.5B",  		
    "DeepSeek-R1-Distill-Qwen-32B",   		
    "DeepSeek-R1-Distill-Qwen-7B",    		
    
    "gemma-3-27b-it",                 		
    "gemma-3-12b-it",                 		
    "gemma-3-4b-it",                  		
    "gemma-3-1b-it",                  		
    
    "llama-3.2-1B-Instruct",          		
    "llama-3.2-3B-Instruct",      
    "llama-3.3-70B-Instruct",
    "llama-4-Scout-17B-16E-Instruct",
    "meta-llama-3.1-8B-Instruct",
    
    "mistral-7B-Instruct-v0.3",
    "mistral-small-3.2-24B-Instruct-2506",
    
    "Phi-4-mini-instruct",
    "phi-4",
    
    "Qwen3-0.6B",
    "Qwen3-14B",
    "Qwen3-1.7B",
    "Qwen3-30B-A3B-Thinking-2507",
    "Qwen3-32B",
    "Qwen3-4B",
    "Qwen3-8B",
    "sabia-7b"
)


llms_no_user_template = ("mistral-7B-Instruct-v0.3")
tp_prompts = ["FS", "ZS", "CoT", "CoT_FS"]

for prompt in tp_prompts:
    dataset_name = f"dataset_{prompt}.csv"
    df=None

    if os.path.isfile(dataset_name): df = pd.read_csv(dataset_name, delimiter=";")
    else: df = pd.read_parquet("dataset_pipeline_sentiment.parquet")

    for LLM_MODEL in llm:
        if LLM_MODEL in df.columns:
            print(f"[{prompt}] {LLM_MODEL} |> Já processado ")
            continue

        client=None
        if LLM_MODEL in ["gpt-5", "gpt-5-nano"]:
            client = OpenAI(api_key=chatgpt_key)
        else:
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        with open(f"templates/{prompt}.txt", 'r', encoding='utf-8') as arquivo:
            content_prompt = arquivo.read()
        
        # Contagem
        correct_predictions = 0
        total_predictions = 0

        # Novas colunas
        preffix_model = LLM_MODEL
        df[preffix_model] = ""
        df[preffix_model+"_response"] = ""

        for index, row in df.iterrows():
            user_message = f"""
                User text: {row['CONTENT']}
                Response:
            """
            start_time = time.time()
            messages = None
            if LLM_MODEL in llms_no_user_template:
                messages=[                    
                    {"role": "user", "content": content_prompt + "\n" + user_message}
                ]
            else:
                messages=[
                    {"role": "system", "content": content_prompt},
                    {"role": "user", "content": user_message}
                ]
            
            attempt = 0
            while attempt <= MAX_RETRIES:
                try:
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        # temperature=0.0,  # Setting temperature to 0 for more deterministic output
                        max_completion_tokens=1000
                    )
                    break
                except Exception as e:
                    attempt += 1
                    print(f"Tentativa {attempt}/{MAX_RETRIES} ({LLM_MODEL}) falhou com erro: {e}")
                    if attempt > MAX_RETRIES:
                        print(f"Inferência falhou após {MAX_RETRIES} tentativas. Encerrando execução.")
                        raise 
                    else:
                        print("Aguardando 5 segundos antes de tentar novamente.")
                        time.sleep(5)

            end_time = time.time()
            duration = end_time - start_time
            predicted_module = ExtractlyCommon.extract_sentiment(completion.choices[0].message.content) 
            print(f"⏱️ Timing  : {duration}")
            print(f"🎲 Dataset : {row['SENTIMENT']}")
            print(f"🤖 IA      : {predicted_module[0] if type(predicted_module) == list else predicted_module}")
            print()

            try:
                df.loc[index, preffix_model] = predicted_module[0] if type(predicted_module) is list else predicted_module
                df.loc[index, preffix_model+"_response"] = completion.choices[0].message.content
                df.loc[index, preffix_model+"_time"] = duration
                df.loc[index, preffix_model+"_n_tokens_in"] = completion.usage.prompt_tokens
                df.loc[index, preffix_model+"_n_tokens_out"] = completion.usage.completion_tokens
            except Exception as ex:
                print(ex)
                continue
            
            print(f"[{LLM_MODEL}/{prompt}] {index + 1} of {len(df)}")
        
        correct_predictions = (df["SENTIMENT"] == df[LLM_MODEL]).sum()
        total_examples = len(df)
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0 # Avoid division by 0
        print(f"Accuracy {LLM_MODEL} [{prompt}]: {accuracy:.2%}")

        dataset_name = f'dataset_{prompt}.csv'
        df.to_csv(dataset_name, index=False, sep=";")