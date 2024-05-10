# -*- coding: utf-8 -*-

# @Time    : 2024/4/10 16:33
# @Author  : kewei

import os
import json


def format_chat_data(res: dict, num_process):
    col_map = {
        '测试日期': res['date'][:8],
        '显卡型号': 'A6000',
        '数量': 1,
        '模型名称': res['model_id'],
        '支持上下文': '4k',
        '请求数量': res['num_prompts'],
        '并发数': num_process,
        '整体耗时(s)': round(res['duration'], 2),
        '成功返回数量': res['completed'],
        '输入tokens': res['total_input_tokens'],
        '输出tokens': res['total_output_tokens'],
        'QPS': round(res['request_inthroughput'], 2),
        '输入吞吐量(tokens/s)': round(res['input_throughput'], 2),
        '输出吞吐量(tokens/s)': round(res['output_throughput'], 2),
        '平均第一次返回时间(ms)': round(float(res['mean_ttft_ms']), 2),
        '中位第一次返回时间(ms)': round(res['median_ttft_ms'], 2),
        'p99第一次返回时间(ms)': round(res['p99_ttft_ms'], 2),
        '平均每个 token(ms)': round(res['mean_tpot_ms'], 2),
        '中位数每个 token(ms)': round(res['median_tpot_ms'], 2),
        'p99 每个token(ms)': round(res['p99_tpot_ms'], 2)
    }
    return col_map


def format_embedding_data(res: dict, num_process):
    col_map = {
        '测试日期': res['date'][:8],
        '显卡型号': 'A6000',
        '数量': 1,
        '模型名称': res['model_id'],
        '支持上下文': '4k',
        '请求数量': res['num_prompts'],
        '并发数': num_process,
        '整体耗时(s)': round(res['duration'], 2),
        '成功返回数量': res['completed'],
        '输入tokens': res['total_input_tokens'],
        'QPS': round(res['request_inthroughput'], 2),
        '输入吞吐量(tokens/s)': round(res['input_throughput'], 2),
        '平均每个 token(ms)': round(res['mean_tpot_ms'], 2),
        '中位数每个 token(ms)': round(res['median_tpot_ms'], 2),
        'p99 每个token(ms)': round(res['p99_tpot_ms'], 2)
    }
    return col_map


def main():
    path = 'output'
    # 读取path下的所有文件
    files = os.listdir(path)
    # 按照时间排序，取最近7个文件
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
    num = 1
    all_chat = []
    all_embedding = []
    for i in range(0, len(files), 7):
        if num > 100:
            break
        num += 1
        files_tmp = files[i: i + 7][::-1]
        for file in files_tmp:
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as f:
                res = json.loads(f.read())
            log_type = file.split('-')[0]
            num_process = file.split('-')[1]
            if log_type == 'chat':
                all_chat.append(format_chat_data(res, num_process))
            elif log_type == 'embedding':
                all_embedding.append(format_embedding_data(res, num_process))

    # save csv
    import pandas as pd
    pd.DataFrame(all_chat).to_csv('output_chat.csv', index=False)
    pd.DataFrame(all_embedding).to_csv('output_embedding.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
