import json
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from transformers import BartTokenizerFast


'''
TJY
train.prune.json -> train.template_1.json, train.bert-base-cased_1.json
train.prune.json -> train.template_2.json, train.bert-base-cased_2.json
train.prune.json -> train.template_3.json, train.bert-base-cased_3.json
train.prune.json -> train.template_4.json, train.bert-base-cased_4.json
train.prune.json -> train.template_5.json, train.bert-base-cased_5.json

train.negative.prune.json -> train.negative.template_1.json, train.negative.bert-base-cased_1.json
train.negative.prune.json -> train.negative.template_2.json, train.negative.bert-base-cased_2.json
train.negative.prune.json -> train.negative.template_3.json, train.negative.bert-base-cased_3.json
train.negative.prune.json -> train.negative.template_4.json, train.negative.bert-base-cased_4.json
train.negative.prune.json -> train.negative.template_5.json, train.negative.bert-base-cased_5.json
'''

tokenizer = BartTokenizerFast.from_pretrained("/root/fsl-proact-main/bart-base", add_special_tokens=True)

# Load the vocab
with open('vocab.pkl', 'rb') as f:
    word_map, _ = pickle.load(f)


def compute_softmax(x):
    # Normalize the input
    exp_x = np.exp(x - np.max(x))  # Preventing overflow
    return exp_x / np.sum(exp_x)


def read_template_multi_file(file_path):
    with open(file_path, 'r') as f_multi:
        template_multi_all = f_multi.readlines()
        template_multi_list = []
        log_likelihood_list = []
        for idx in range(4, 7):  # 4-7: 3 samples, 4-8: 4 samples, 4-9: 5 samples, 4-10: 6 samples
            template_ll = template_multi_all[idx]
            template, log_likelihood = template_ll.strip('\n').split('\t')
            template_multi_list.append(template)
            log_likelihood_list.append(float(log_likelihood))
    log_likelihood_arr = np.array(log_likelihood_list)
    log_likelihood_arr = compute_softmax(log_likelihood_arr)
    return template_multi_list, log_likelihood_arr


def generate_template(input_path, template_multi_list):
    with open(input_path, "r") as file:
        datas = json.load(file)
        new_data_items = []
        tokenized_items = []
        max_token_len = 0  # Maximum token length
        max_prompt_len = 0  # Maximum prompt length
        token_len_list = []  # Record the token length of each sentence in a list
        prompt_len_list = []  # Record the token length of each prompt in a list
        for data_id, data in enumerate(datas):
            # Load information
            input_token = data['token']
            input_trigger = data['trigger']
            input_argument = data['argument']

            # 遍历所有的模板
            all_input_prompt_lists = []
            for tem_id, template in enumerate(template_multi_list):
                # load prompt template
                template_split = template.strip().split('*')
                input_prompt_list = []
                for one_split in template_split:
                    if one_split == 'cls' or one_split == 'sentu_0' or one_split == 'sep+' or one_split == 'mask' or one_split == '':
                        continue
                    elif one_split == 'trigger':
                        for trigger_one in input_token[input_trigger[0]:input_trigger[1] + 1]:
                            input_prompt_list.append(trigger_one)
                    elif one_split == 'argument':
                        argument_set = set(argument[2] for argument in input_argument if argument[2] != 'Place' and 'Time' not in argument[2])
                        if len(argument_set) == 0:
                            argument_set.add('None')
                        for argument_one in argument_set:
                            input_prompt_list.append(argument_one)
                    else:
                        input_prompt_list.append(one_split.strip('_'))
                all_input_prompt_lists.append(input_prompt_list)

            # Establish the new data
            new_data = copy.deepcopy(data)

            id_dict = {}
            for tem_id, input_prompt_list in enumerate(all_input_prompt_lists):
                # 输入的prompt
                prompt_input = " ".join(input_prompt_list) + ' [MASK]'
                prompt_token = tokenizer(prompt_input)
                prompt_input_ids, prompt_mask_ids = prompt_token["input_ids"], prompt_token["attention_mask"]
                new_data['tem_{}_mask_id'.format(str(tem_id))] = len(prompt_input_ids) - 1
                id_dict['tem_{}_prompt_input_ids'.format(str(tem_id))] = prompt_input_ids
                id_dict['tem_{}_prompt_mask_ids'.format(str(tem_id))] = prompt_mask_ids
                prompt_len = len(prompt_input_ids)
                prompt_len_list.append(prompt_len)
                if prompt_len > max_prompt_len:
                    max_prompt_len = prompt_len

            sentence_input = " ".join(input_token[:input_trigger[0]]) + ' <t> ' \
                             + " ".join(input_token[input_trigger[0]:input_trigger[0] + 1]) + ' </t> ' \
                             + " ".join(input_token[input_trigger[0] + 1:])

            sentence_token = tokenizer(sentence_input)
            sentence_input_ids, sentence_mask_ids = sentence_token["input_ids"], sentence_token["attention_mask"]
            new_data['trigger'] = [input_trigger[0] + 1, input_trigger[1] + 1]  # The trigger id should also be updated
            id_dict['sentence_input_ids'] = sentence_input_ids
            id_dict['sentence_mask_ids'] = sentence_mask_ids
            id_dict['id'] = data['id']

            # Append a new data to new_data_items
            new_data_items.append(new_data)
            # Append a new line to tokenized_items
            tokenized_items.append(id_dict)

            # Count sentence length
            token_len = len(sentence_input_ids)
            token_len_list.append(token_len)
            if token_len > max_token_len:
                max_token_len = token_len

    # Save the template file and bert indices file
    dataset_name, _, file_name = input_path.split('/')
    new_template_name = file_name.replace('prune', 'template')
    new_bert_name = file_name.replace('prune', 'bert-base-cased')
    with open(dataset_name + '/template_argument_multi_bart/' + new_template_name, 'w') as json_file:
        json.dump(new_data_items, json_file, indent=4)
    with open(dataset_name + '/template_argument_multi_bart/' + new_bert_name, 'w') as json_file:
        json.dump(tokenized_items, json_file, indent=4)

    # Show the statistics of sentence length
    print('{}的最大句子长度为：{}，最大提示长度为：{}'.format(input_path, max_token_len, max_prompt_len))
    token_len_arr = np.array(token_len_list)
    toekn_len_arr_sorted = np.sort(token_len_arr)  # Sorting data
    hist_data, bin_edges = np.histogram(toekn_len_arr_sorted, bins=10)  # Calculate the distribution of the data (histogram)
    plt.bar(range(len(hist_data)), hist_data, align='center')  # Plotting bar charts
    plt.xticks(range(len(hist_data)),
               ['{:.0f}-{:.0f}'.format(bin_edges[i], bin_edges[i + 1]) for i in range(len(hist_data))],
               rotation=45)  # Setting the x-axis scale labels

    # Setting the chart title and axis labels
    plt.title('Data Distribution Bar Chart')
    plt.xlabel('Data Range')
    plt.ylabel('Frequency')

    # Show Chart
    plt.show()


if __name__ == '__main__':
    dataset_name = 'rams'  # rams, ace, lowkbp0, lowkbp1, lowkbp2, lowkbp3, lowkbp4
    template_multi_path = dataset_name + "/template_argument_multi_bart/template_argument_20230522.txt"
    template_multi_list, log_likelihood_arr = read_template_multi_file(template_multi_path)

    # Save the likelihood of the template
    # Open the txt file and create it if it doesn't exist
    ll_path = dataset_name + "/template_argument_multi_bart/template_log_likelihood.txt"
    with open(ll_path, 'w') as f:
        # Write text content to txt file
        log_likelihood_str = '\t'.join([str(ll) for ll in log_likelihood_arr])
        f.write(log_likelihood_str)

    # Save Template
    generate_template(dataset_name + '/fsl/train.prune.json', template_multi_list)
    generate_template(dataset_name + '/fsl/dev.prune.json', template_multi_list)
    generate_template(dataset_name + '/fsl/test.prune.json', template_multi_list)
    generate_template(dataset_name + '/fsl/train.negative.prune.json', template_multi_list)  # lowkbp has no negative samples
    generate_template(dataset_name + '/fsl/dev.negative.prune.json', template_multi_list)  # lowkbp has no negative samples
    generate_template(dataset_name + '/fsl/test.negative.prune.json', template_multi_list)  # lowkbp has no negative samples
