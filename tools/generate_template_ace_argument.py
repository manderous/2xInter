import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import torch
import os
from tqdm import tqdm
import json
import argparse
import pandas as pd

def get_text(template, input_text_tuple, trigger, argument, label, tokenizer):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)
    special_token_mapping = {'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id}
    for i in range(10):
        special_token_mapping["<extra_id_%d>" % (i)] = tokenizer._convert_token_to_id("<extra_id_%d>" % (i))
    template_list = template.split('*')
    input_ids = []
    for part in template_list:
        new_tokens = []
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
        elif part[:5] == 'label':  # tjy:不需要mapping了
            label_tokens = []
            label_tokens.append(enc(' ' + label.split(':')[0]))
            for label_suffix in label.split(':')[1].split('-'):
                label_tokens.append(enc(label_suffix))
            label_tokens = [item for sublist in label_tokens for item in sublist]  # 将列表中的小列表合并成一个大列表
            new_tokens += label_tokens
        elif part[:8] == 'argument':  # tjy:增加论元
            argument_tokens = []
            for argument_one in argument.split('/'):
                argument_tokens.append(enc(argument_one))
            argument_tokens = [item for sublist in argument_tokens for item in sublist]  # 将列表中的小列表合并成一个大列表
            new_tokens += argument_tokens
        elif part[:7] == 'trigger':  # tjy:增加触发词
            new_tokens += enc(' ' + trigger)
        elif part[:5] == 'sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_tuple[sent_id])
        elif part[:6] == '+sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(' ' + input_text_tuple[sent_id]) # add space
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_tuple[sent_id][:-1])
        elif part[:7] == '+sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:7] == '+sentu_':
            # Upper case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:6] == 'sentu_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == 'sentl-_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        else:
            part = part.replace('_', ' ') # there cannot be space in command, so use '_' to replace space
            # handle special case when t5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += enc(part)

        input_ids += new_tokens
    return input_ids

def generate(dataset, template, model, tokenizer, target_number, beam, label=None, length_limit=None, truncate=None):
    """
    Generate templates based on given inputs

    label: Only use instances with this label (deprecated)
    length_limit: At least generate content as long as length_limit (deprecated)
    """
    input_texts = []
    input_tensors = []
    max_length = 0

    # Process the inputs
    for item in dataset:
        if label is None or item['label'] == label:
            input_text = get_text(template, item['text'], item['trigger'], item['argument'], item['label'], tokenizer)
            if truncate is not None:
                if truncate == 'head':
                    input_text = input_text[-256:]
                elif truncate == 'tail':
                    input_text = input_text[:256]
                else:
                    raise NotImplementedError
            input_ids = torch.tensor(input_text).long()
            max_length = max(max_length, input_ids.size(-1))
            input_tensors.append(input_ids)

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, :input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, :input_tensors[i].size(-1)] = 1

    # Print some examples
    print('####### example #######')
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(input_ids[1]))
    print(tokenizer.decode(input_ids[2]))
    print('####### example #######\n')

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0

    # Maximum generate content length
    max_length = 20

    start_mask = tokenizer._convert_token_to_id('<extra_id_0>')
    ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
    ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

    # decoder_input_ids: decoder inputs for next regressive generation
    # ll: log likelihood
    # output_id: which part of generated contents we are at
    # output: generated content so far
    # last_length (deprecated): how long we have generated for this part
    current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
    for i in tqdm(range(max_length - 2)):
        new_current_output = []
        for item in current_output:
            if item['output_id'] > target_number:
                # Enough contents
                new_current_output.append(item)
                continue
            decoder_input_ids = item['decoder_input_ids']

            # Forward
            batch_size = 32
            turn = input_ids.size(0) // batch_size
            if input_ids.size(0) % batch_size != 0:
                turn += 1
            aggr_output = []
            for t in range(turn):
                start = t * batch_size
                end = min((t + 1) * batch_size, input_ids.size(0))

                with torch.no_grad():
                    aggr_output.append(model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.cuda()[start:end])[0])
            aggr_output = torch.cat(aggr_output, 0)

            # Gather results across all input sentences, and sort generated tokens by log likelihood
            aggr_output = aggr_output.mean(0)
            log_denominator = torch.logsumexp(aggr_output[i], -1).item()
            ids = list(range(model.config.vocab_size))
            ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
            ids = ids[:beam+3]
            
            for word_id in ids:
                output_id = item['output_id']

                if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('</s>'):
                    # Finish one part
                    if length_limit is not None and item['last_length'] < length_limit[output_id - 1]:
                        check = False
                    else:
                        check = True
                    output_id += 1
                    last_length = 0
                else:
                    last_length = item['last_length'] + 1
                    check = True

                output_text = item['output'] + [word_id]
                ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                new_decoder_input_ids[:] = decoder_input_ids
                new_decoder_input_ids[..., i + 1] = word_id

                # Forbid single space token, "....", and ".........."
                if word_id in [3, 19794, 22354]:
                    check = False

                # Forbid continuous "."
                if len(output_text) > 1 and output_text[-2] == 5 and output_text[-1] == 5:
                    check = False

                if check:
                    # Add new results to beam search pool
                    new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                    new_current_output.append(new_item)

        if len(new_current_output) == 0:
            break

        new_current_output.sort(key=lambda x: x['ll'], reverse=True)
        new_current_output = new_current_output[:beam]
        current_output = new_current_output

    result = []
    result_ll = []
    print("####### generated results #######")
    for item in current_output:
        generate_text = ''
        for token in item['output']:
            generate_text += tokenizer._convert_id_to_token(token)
        print('--------------')
        print('score:', item['ll'].item())
        print('generated ids', item['output'])
        print('generated text', generate_text)
        result.append(generate_text)
        result_ll.append(float(item['ll']))
    print("####### generated results #######\n")

    return result, result_ll

def load_dataset(data_dir):
    lines = open(os.path.join(data_dir, 'train_argument.tsv')).readlines()
    dataset = []
    for line in lines:
        line = line.strip().split('\t')
        dataset.append({'text': [line[0]], 'trigger': line[1], 'argument': line[2], 'label': line[3]})
    return dataset

def search_template(model, tokenizer, task_name, k, seed, beam, output_dir, data_dir):
    print('#', task_name, k, seed, beam)
    dataset_path = os.path.join(data_dir, task_name)
    dataset = load_dataset(dataset_path)
    print('|', 'dataset examples')
    print('|', dataset[0])
    print('|', dataset[-1])
    print()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
    f = open(os.path.join(output_dir, task_name, "template.txt"), 'w')

    # ACE-Single sentence tasks
    # We take two kinds of templates: put [MASK] at the beginning or the end
    # template = "*cls**sentu_0**<extra_id_0>**label**<extra_id_1>**sep+*"
    # template = "*cls**sentu_0**<extra_id_0>**trigger**<extra_id_1>**label**<extra_id_2>**sep+*"
    # template = "*cls**sentu_0**<extra_id_0>**trigger**<extra_id_1>**label**sep+*"
    template = "*cls**sentu_0**trigger**<extra_id_0>**argument**<extra_id_1>**label**sep+*"
    generate_text, text_lls = generate(dataset, template, model, tokenizer, target_number=2, beam=beam, label=None,truncate='head')[:beam // 2]

    print("####### generated templates #######")
    for text, text_ll in zip(generate_text, text_lls):
        # Transform T5 outputs to our template format
        text = text.replace('<extra_id_0>', '*cls**sentu_0**trigger*')
        text = text.replace('<extra_id_1>', '*argument*')
        text = text.replace('<extra_id_2>', '*mask**sep+*')
        text = text.replace('</s>', '*mask**sep+*')
        text = text.replace('▁', '_')
        print('text: ', text, ' log likelihood: ', text_ll)
        f.write(text + '\t' + str(text_ll) + '\n')
    print("####### generated templates #######\n")

    # template = "*cls*.*<extra_id_0>**label**<extra_id_1>**+sentu_0**sep+*"
    # template = "*cls**<extra_id_0>**trigger**<extra_id_1>**label**<extra_id_2>**+sentu_0**sep+*"
    # template = "*cls**<extra_id_0>**trigger**<extra_id_1>**label**+sentu_0**sep+*"
    template = "*cls**trigger**<extra_id_0>**argument**<extra_id_1>**label**+sentu_0**sep+*"
    generate_text, text_lls = generate(dataset, template, model, tokenizer, target_number=2, beam=beam, label=None, truncate='tail')[:beam // 2]
    print("####### generated templates #######")
    for text, text_ll in zip(generate_text, text_lls):
        # Transform T5 outputs to our template format
        text = text.replace('<extra_id_0>', '*cls**trigger*')
        text = text.replace('<extra_id_1>', '*argument*')
        text = text.replace('<extra_id_2>', '*mask**+sent_0**sep+*')
        text = text.replace('</s>', '*mask**+sent_0**sep+*')
        text = text.replace('▁', '_')
        print('text: ', text, ' log likelihood: ', text_ll)
        f.write(text + '\t' + str(text_ll) + '\n')
    print("####### generated templates #######\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_model', type=str, default='t5-3b', help='T5 pre-trained model')
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 13, 21, 100, 87], help="Data split seeds")
    parser.add_argument('--task_name', type=str, nargs='+', default=['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'], help="Task names")
    parser.add_argument('--output_dir', type=str, default='Output directory')

    # parser.add_argument('--data_dir', type=str, default="data/k-shot", help="Data directory")
    parser.add_argument('--data_dir', type=str, default="./datasets/k-shot", help="Data directory")  # tjy
    parser.add_argument('--beam', type=int, default=100, help="Beam search width")
    parser.add_argument('--k', type=int, default=16, help="Number of training instances per label")
 
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    tokenizer.sep_token = '</s>'

    model = model.cuda()
    model.eval()

    for task_name in args.task_name:
        for seed in args.seed:
            search_template(model=model, tokenizer=tokenizer, task_name=task_name, k=args.k, seed=seed, beam=args.beam, output_dir=args.output_dir, data_dir=args.data_dir)

if __name__ == '__main__':
    main()