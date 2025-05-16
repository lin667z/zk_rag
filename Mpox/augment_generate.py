import json
from openai import OpenAI
# from tqdm import tqdm

num=0

def get_gpt_response_w_system(prompt, options):
    # global system_prompt            # 没用到，浪费资源
    s_prompt = options['system_prompt'] # prompt，提示词
    base_url = options['gpt-baseurl']   # api的网址
    api_key = options['gpt-apikey']     # api的apikey
    model = options['chat-model']       # chatgpt类型
    try:
        client = OpenAI(
        base_url=base_url,
        api_key=api_key
        )
        completion = client.chat.completions.create(
        temperature= 0,
        model=model,
        messages=[
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": prompt}
        ]
        )
        content = completion.choices[0].message.content
        return content
    except Exception as e:
        print(e)
        return content

def knowledge2str(knowledge, tokens_per_knowledge):
    knowledgeStr = ''
    for document in knowledge:
        # meta = document.metadata
        # 知识过长直接截断
        if(len(document.page_content) > tokens_per_knowledge):
            document.page_content = document.page_content[:tokens_per_knowledge]

        knowledgeStr += document.page_content
    return knowledgeStr


def generate_answer(query, knowledge, options):
    prompt = ""
    # prompt = query
    # prompt += '你可能会用到以下知识：'
    # prompt += knowledge2str(knowledge, options['tokens_per_knowledge'])

    # 自己改的，提升性能
    word = [query,'你可能会用到以下知识：',knowledge2str(knowledge, options['tokens_per_knowledge'])]
    prompt = '\n'.join(word)

    # with open(sys_path, 'r') as f:
    #     for line in f.readlines():
    #         system_prompt += line

    # retrieval_prompt = ""
    # with open(aug_path, 'r') as f:
    #     for line in f.readlines():
    #         retrieval_prompt += line

    response = get_gpt_response_w_system(prompt, options)
    return response