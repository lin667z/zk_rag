from Mpox.read_from_db import  read_from_db
from augment_generate import generate_answer
from tqdm import tqdm  #进度条
from Mpox.score import get_score
from Mpox.utils import write_csv, calculate_avg, count_lines_in_jsonl, write_jsonl, read_jsonl

options = dict()
# 可能影响性能
options['k'] = 9     # 返回相似文档的数量
options['system_prompt'] = ('你是一个熟悉猴痘专业医生，\
                             擅长使用权威性的猴痘数据、文档的知识来回答医生，\
                             针对用户的提问，你会得到一些知识辅助，请忽略没有帮助的知识，\
                             结合有用的部分以及你的知识，尽可能权威地用英语回答用户，并给出出处和网址\
                             并再出处前标明“source:”，网址前标明“url:”'
                            )
# ‘尽可能’不确定有无影响
# options['system_prompt'] = '你是一个蚂蚁集团的TuGraph数据库专家，\
#                             擅长使用TuGraph数据库的相关知识来回答用户的问题，\
#                             针对用户的提问，你会得到一些知识辅助，请忽略没有帮助的知识，\
#                             结合有用的部分以及你的知识，尽可能简洁地直接给出答案，不需要任何解释。\
#                             注意：问题中的数据库一律指代TuGraph,问及系统是否支持某些功能时，若不清楚一律回答暂不支持\
#                             请仿照下面的样例答案格式进行后续的回答：\
#                             样例问题1："RPC 及 HA 服务中，verbose 参数的设置有几个级别？", 样例答案: "三个级别（0，1，2）。"\
#                             样例问题2:"如果成功修改一个用户的描述，应返回什么状态码？"样例答案：“200” '
options['chat-model'] = "gpt-4o-mini"
options['embedding-model'] = "text-embedding-3-large"
options['tokens_per_knowledge'] = 2000 # 为防止单个知识过长，进行截断
# gpt调用
options['gpt-baseurl'] = ''
options['gpt-apikey'] = ""
# 文件路径
options['persist_directory'] = './db/xldatabase/rag'
options['test_path'] = './test/test1.jsonl'
options['val_path'] = './test/val.jsonl'
# 输出路径
options['test_out_path'] = './result/answer_test.jsonl' 
options['val_out_path'] = './result/answer_val.jsonl'
options['score_path'] = './result/score.csv'
options['retrieval_path'] = './result/' # 对检索得到的知识输出
# 功能开启，1表示开启
options['use_val'] = 0
options['use_val_score'] = 0
options['use_test'] = 0
options['save_knowledge'] = 0 # 把问题对应检索知识保存下来


if options['use_val']:
    print('正在对 val.jsonl 进行生成检索.....')
    answers_val = []
    if options['save_knowledge']: #是否要保存知识
        knowledge_val = []
    with tqdm(total=count_lines_in_jsonl(options['val_path'])) as pbar:
        for obj in read_jsonl(options['val_path']):
            query = obj.get('input_field') # 获取提问
            if options['save_knowledge']:
                knowledges = read_from_db(query, options['k'], options) # 读取知识a list of Documents
                # 保存知识
                knowledge_val.append(dict(Q = query, K1 = knowledges[0], K2 = knowledges[1], K3 = knowledges[2]))
                # 生成答案
                answers_val.append(dict(id=obj.get('id'), output_field = generate_answer(query,knowledges, options)))
            else:
            # 生成答案
                answers_val.append(dict(id=obj.get('id'), output_field = generate_answer(query, read_from_db(query, options['k'], options), options)))
            pbar.update(1)
    # 答案写入文件
    write_jsonl(answers_val, options['val_out_path'])
    if options['save_knowledge']:
        # 知识保存写入文件
        write_csv(knowledge_val, options['retrieval_path']+ 'retrieval_val.csv')
    print('val.jsonl 已生成答案！\n \n')

if options['use_val_score']:
    print('正在计算分数.....')
    # 计算得分
    score_output = get_score(options)
    # 写入文件
    write_csv(score_output, options['score_path'])
    print('分数平均为{}! \n \n'.format(calculate_avg(score_output)))
    # write_jsonl(score_output, options['score_path'])

if options['use_test']:
    print('正在对 test1.jsonl 进行生成检索.....')
    answers_test = []
    if options['save_knowledge']: # 知识保存
        knowledge_test = []
    with tqdm(total=count_lines_in_jsonl(options['test_path'])) as pbar:
        for obj in read_jsonl(options['test_path']):
            query = obj.get('input_field')
            if options['save_knowledge']:
                # 知识检索和答案生成
                knowledges = read_from_db(query, options['k'], options) # a list of Documents
                knowledge_test.append(dict(Q = query, K1 = knowledges[0], K2 = knowledges[1], K3 = knowledges[2]))
                answers_test.append(dict(id=obj.get('id'), output_field = generate_answer(query,knowledges, options)))
            else:
                # 生成问题答案
                answers_test.append(dict(id=obj.get('id'), output_field = generate_answer(query, read_from_db(query, options['k'], options), options)))
            pbar.update(1)
    # 结果写入文件
    write_jsonl(answers_test, options['test_out_path'])
    if options['save_knowledge']:
        write_csv(knowledge_test, options['retrieval_path']+ 'retrieval_test.csv')
    print('test1.jsonl 已生成答案！\n \n')


