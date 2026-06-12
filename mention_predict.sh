#!/bin/bash

# cd /home/zhangzy/CLllm/src/rag_emc && python run_pipeline_multi_env.py \
#     --data_file_path /home/zhangzy/CLllm/src/get_mention/converted_mention_result.json \
#     --retrieval_output_path retrieval/candidate_mentions_result_bc5cdr_test_retrieval_results.json \
#     --validation_output_path validation_output/candidate_mentions_result_bc5cdr_test_validation_results.json \
#     --log_name validation_output/candidate_mentions_result_bc5cdr_test.log


cd /home/zhangzy/CLllm/src/rag_emc && python g_main.py \
    --retrieval_input_path retrieval_output/candidate_mentions_result_bc5cdr_test_retrieval_results.json \
    --validation_output_path validation_output/candidate_mentions_result_bc5cdr_test_validation_results.json
