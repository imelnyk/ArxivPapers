import tiktoken
import os
import time
import re
import json
from nltk.tokenize import sent_tokenize
import pyperclip


def gpt_short_verbalizer(files_dir, llm_api, llm_strong, llm_base, logging):

    encoding = tiktoken.get_encoding("cl100k_base")

    path = os.path.join(files_dir, "extracted_orig_text_clean.txt")
    with open(path) as f:
        paper_text = f.read()

    skip_words = ['introduction', 'related', 'notation', 'literature']
    stop_words = ['references', 'appendix', 'conclusion', 'experiment', 'discussion', 'acknowledgments', 'about this document']
    sections = paper_text.split('Section: ### ')
    good_text = ''
    for s in sections:
        splits = s.split(' ###. ')
        if len(splits) != 2:
            continue
        title, main_text = splits

        should_skip = False
        for w in skip_words:
            if w in title.lower():
                should_skip = True
                break

        if should_skip:
            continue

        should_stop = False
        for w in stop_words:
            if w in title.lower():
                should_stop = True
                break

        if should_stop:
            break

        good_text += title + '. ' + main_text

    max_tokens = 14000
    while(len(encoding.encode(good_text)) > max_tokens):
        extra = max_tokens - len(encoding.encode(paper_text))
        good_text = good_text[:-(extra + 10)]

    human_message = 'you are student who wrote this paper. ' \
                    'you are in elevator with your friend and you have only about ' \
                    '1 minute worth of text to describe your work. ' \
                    'What is the main idea here? What exactly did author propose? What is their main algorithm? ' \
                    f'Show step 1, step 2, step 3, step 4, and so on <<{good_text}>>.'

    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': human_message}
    ]

    for i in range(3):
        try:
            response = llm_api(model=llm_base, messages=messages, temperature=0)
            break
        except:
            time.sleep(5)
    else:
        raise Exception(f"{llm_base} failed")

    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    prompt = ("for each bullet, show specifics. if improved something, "
              "show by how much, if modified something, show what exactly, "
              "if achieved something, specify what exactly.")
    messages.append({"role": "user", "content": prompt})

    for i in range(3):
        try:
            response = llm_api(model=llm_base, messages=messages, temperature=0)
            break
        except:
            time.sleep(5)
    else:
        raise Exception(f"{llm_base} failed")

    prompt = 'convert text denoted by double angle brackets to a cohesive and educational text of up to 300 words'  \
             f' and at most 5 paragraphs. <<{response.choices[0].message.content}>>. '  \
             'Important: Never say that results or details are not provided. Never use any latex or '  \
             'math expressions. Start with "In this paper, we" and continue in first person plural point of view.'

    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt}
    ]

    for i in range(3):
        try:
            response = llm_api(model=llm_strong, messages=messages, temperature=0)
            break
        except:
            time.sleep(5)
    else:
        raise Exception(f"{llm_strong} failed:")

    gpttext = response.choices[0].message.content

    logging.info(f'Short GPT Summary: \n{gpttext}\n')
    logging.info("-" * 100)

    gpttext = gpttext.replace("$", '').replace("```", '').replace("<<", '').replace(">>", '').replace("**", '')

    # remove words with underscores
    gpttext = re.sub(r'\b\w*__\w*\b', '', gpttext)

    prompt = 'create a set of slides each with title and 3-4 bullet points, reflecting main idea in this text. ' \
             'Use one slide for each paragraph. Title is short and representative (at most 2 words). ' \
             'Bullet points are short, concise and to the point (at most 4 words). Text to extract the slides are' \
             f' inside double angle brackets <<{response.choices[0].message.content}>>. ' \
             'Never say that results or details are not provided. Ensure that you assign one slide per paragraph.'

    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt}
    ]


    for i in range(3):
        try:
            response = llm_api(model=llm_strong,
                               messages=messages,
                               temperature=0,
                               functions=[
                               {
                                   "name": "create_slides",
                                   "description": "create slides from title and a list of bullets",
                                   "parameters": {
                                       "type": "object",
                                       "properties": {
                                           "slides": {
                                               "type": "array",
                                               "description": "the list of slides",
                                               "items": {
                                                   "type": "array",
                                                   "description": "individual slide, first element title, rest-bullet points",
                                                   "items": {
                                                       "type": "string",
                                                       "description": "first item - title; all other items - bullet points",
                                                   },
                                               }
                                           },
                                       },
                                       "required": ["slides"],
                                   },
                               }
                              ],
                              function_call="auto")
            break
        except:
            time.sleep(5)
    else:
        raise Exception(f"{llm_strong} function call failed")

    slides = json.loads(response.choices[0].message.function_call.arguments)

    return gpttext, slides


def gpt_qa_verbalizer(files_dir, llm_api, llm_base, matcher, logging):

    encoding = tiktoken.get_encoding("cl100k_base")

    path = os.path.join(files_dir, "original_text_split_pages.txt")
    with open(path) as f:
        paper_text = f.read()

    system_message = 'You are a college professor, known for your expert knowledge in deep learning field. ' \
                     'You are also known for creating very thoughtful and probing questions that examine' \
                     'the actual knowledge of a student based on their submitted paper. Your goal is to come up with ' \
                     'a list of questions, both on intuitive level and on deeper technical level that evaluate if ' \
                     'a student really knows about his or her work. Focus on the knowledge of the main proposed method, ' \
                     'motivation and results. Make sure your list of questions examine the student thoroughly. ' \
                     'Ask at least 10 different and diverse questions. ' \
                     'The questions must cover intuition, main idea and technical details, among others. ' \
                     'Be extremely specific and ask about details presented in the paper, no generic or abstract questions. '

    human_message = f'Below is the student arxiv paper about which the questions needs to be asked: {paper_text}'

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': human_message}
    ]

    for i in range(3):
        try:
            response = llm_api(model=llm_base,
                               messages=messages,
                               temperature=0,
                               functions=[
                                   {
                                       "name": "ask_questions",
                                       "description": "ask questions about provided arxiv paper",
                                       "parameters": {
                                           "type": "object",
                                           "properties": {
                                               "questions": {
                                                   "type": "array",
                                                   "description": "the list of questions to be asked",
                                                   "items": {
                                                       "type": "string",
                                                       "description": "individual question, thoughtful and revealing",
                                                   }
                                               },
                                           },
                                           "required": ["questions"],
                                       },
                                   }
                               ],
                               function_call="auto",
                               )
            break
        except:
            time.sleep(5)
    else:
        raise Exception(f"{llm_base} failed")

    Qs = json.loads(response.choices[0].message.function_call.arguments)

    system_message = 'You are a student, who wrote this paper. You are on a very important exam. ' \
                     'You are tasked to explain your work as best as you can. ' \
                     'You will be provided with a text of the paper, split by pages and a question. ' \
                     'You must answer the question using information given in the paper. ' \
                     'The answer should be consice and to the point but still contain details. ' \
                     'And it should answer the question as best as possible. Be extremly specific. ' \
                     'Ground your response to the provided paper text. Do NOT use generic or abstract phrases. ' \
                     'Your career depends on how well you do this job. I will tip you $2000 for an excellent job done. ' \
                     'Make sure to answer using at least 10 (ten) sentences.'

    answers = []
    pages = []

    for Q in Qs['questions']:
        human_message = f'Here is the text of the split by pages: {paper_text}. And here is the question you need to answer: {Q}. ' \
                        'Make sure your answer best reflects the provided text.'

        messages = [{'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': human_message}]

        for i in range(3):
            try:
                response = llm_api(model=llm_base, messages=messages, temperature=0)
                break
            except:
                time.sleep(5)
        else:
            raise Exception(f"{llm_base} failed")

        answer = response.choices[0].message.content

        answer = answer.replace("$", '').replace("```", '').replace("<<", '').replace(">>", '').replace("**", '')

        # remove words with underscores
        answer = re.sub(r'\b\w*__\w*\b', '', answer)

        answers.append(answer)

        T = paper_text.split('PAGE ')
        G = answer.split('.')
        seq = matcher.match(G[:-1], T[1:], minilm=1, bert=1, fuzz=1, spacy=1, diff=1, tfidf=1, pnt=True)
        # counts = np.bincount(seq)
        # pages.append(np.argmax(counts))
        pages.append(seq)

    return Qs['questions'], answers, pages


def gpt_textvideo_verbalizer(text, llm_api, llm_strong, llm_base, manual, include_summary, pageblockmap, matcher, logging):

    encoding = tiktoken.get_encoding("cl100k_base")

    # split text into sections
    splits = sent_tokenize(text)

    curr = ''
    curr_upd = []
    gpttext_all = ''
    gptpagemap = []
    textpagemap = []
    page_inds = []
    verbalizer_steps = []

    # split in sections by looking for "Sections: ###" in each sentence produced by sent_tokenize
    sections_split = []
    pagemap_sections = []
    for index, t in enumerate(splits):
        if 'Section: ###' in t:
            sections_split.append([t.split('Section: ')[1]])
            pagemap_sections.append([pageblockmap[index]])
        else:
            sections_split[-1].append(t)
            pagemap_sections[-1].append(pageblockmap[index])

    sections = []
    for sec in sections_split:
        sections.append(' '.join(sec))

    for i_s, sec in enumerate(sections):

        # if there is a mismatch, do this hack to make them of equal length
        if len(sent_tokenize(sec)) != len(pagemap_sections[i_s]):
            minN = min(len(sent_tokenize(sec)), len(pagemap_sections[i_s]))
            cleaned_sent_tok = sent_tokenize(sec)[:minN]
            cleaned_pmap_sec = pagemap_sections[i_s][:minN]
        else:
            cleaned_sent_tok = sent_tokenize(sec)
            cleaned_pmap_sec = pagemap_sections[i_s]

        page_inds += cleaned_pmap_sec

        curr += sec

        for c in cleaned_sent_tok:
            curr_upd.append(c.replace("###.", ' ').replace("###", ''))

        if i_s < len(sections) - 1 and len(encoding.encode(curr)) < 1000:
            continue

        result = re.search(r'###(.*?)###', curr)
        if result:
            sectionName = result.group(1)
        else:
            sectionName = ''

        sectionName = sectionName.rstrip()
        if sectionName.endswith('.'):
            sectionName = sectionName[:-1]

        # curr_upd = ' '.join(curr_upd)
        # curr_upd = curr.replace("###.", '').replace("###", '')

        curr_upto_upto4k = ' '.join(curr_upd)
        while len(encoding.encode(curr_upto_upto4k)) > 3800:
            curr_upto_upto4k = curr_upto_upto4k[:-100]

        sys_message_func =  f"You are an ArXiv paper audio paraphraser. Your primary goal is to " \
                                              "rephrase the original paper content while preserving its overall " \
                                              "meaning and structure, but simplifying along the way, and make it " \
                                              "easier to understand. In the event that you encounter a " \
                                              "mathematical expression, it is essential that you verbalize it in " \
                                              "straightforward nonlatex terms, while remaining accurate, and " \
                                              "in order to ensure that the reader can grasp the equation's " \
                                              "meaning solely through your verbalization. Do not output any long " \
                                              "latex expressions, summarize them in words."


        human_message = 'The text must be written in the first person plural point of view. Do not use long latex ' \
                        'expressions, paraphrase them or summarize in words. Be as faithful to the given text as ' \
                        'possible. Below is the section of the paper, requiring paraphrasing and simplification ' \
                        f' and it is indicated by double angle brackets <<{curr_upto_upto4k}>>. Start with "In this ' \
                        'section, we" and continue in first person plural point of view.'

        messages = [
            {'role': 'system', 'content': sys_message_func},
            {'role': 'user', 'content': human_message}
        ]

        logging.info(f'Section: {sectionName}\n')
        logging.info(sys_message_func + ' ' + human_message)
        logging.info("-" * 100)

        if manual:
            pyperclip.copy(sys_message_func + ' ' + human_message)
            val = input('\nEnter GPT output:')
            if val == '-1':
                break
            gpttext = pyperclip.paste()
            num_input_tokens = len(encoding.encode(sys_message_func + ' ' + human_message))
            num_output_tokens = len(encoding.encode(gpttext))
        else:
            for i in range(3):
                try:
                    response = llm_api(model=llm_strong, messages=messages, temperature=0)
                    gpttext = response.choices[0].message.content
                    num_input_tokens = response.usage.prompt_tokens
                    num_output_tokens = response.usage.completion_tokens
                    break
                except:
                    time.sleep(5)
            else:
                raise Exception(f"{llm_strong} failed")

        gpttext = gpttext.replace("$", '').replace("```", '').replace("<<", '').replace(">>", '').replace("**", '')
        # remove words with underscores
        gpttext = re.sub(r'\b\w*__\w*\b', '', gpttext)

        # replace double newlines with newline
        gpttext = re.sub('\n+', '\n', gpttext)

        addedtext = f' Section: {sectionName}. ' + gpttext + ' '

        smry = ''
        if include_summary:
            sys_message = "As an AI specializing in summarizing ArXiv paper sections, your main task is to distill complex " \
                          "scientific concepts from a given section of a paper into 2-3 simple, yet substantial, " \
                          "sentences. Retain key information, deliver the core idea, and ensure the summary is easy " \
                          "to understand, while not losing the main essence of the content. "

            human_message = 'Below is the section that needs to be summarized in at most 2-3 sentences and it is ' \
                            f'indicated by double angle brackets <<{curr_upto_upto4k}>>. Start with "In this ' \
                            'section, we" and continue in first person plural point of view.'

            messages = [
                {'role': 'system', 'content': sys_message},
                {'role': 'user', 'content': human_message}
            ]

            for i in range(3):
                try:
                    response = llm_api(model=llm_base, messages=messages, temperature=0)
                    summary = response.choices[0].message.content
                    break
                except:
                    time.sleep(5)
            else:
                raise Exception(f"{llm_base} failed")

            summary = summary.replace("$", '').replace("```", '').replace("<<", '').replace(">>", '').replace("**", '')

            smry = ' Section Summary: ' + summary + ' '

            logging.info(f"GPT summary: \n\n{summary}\n")
            logging.info("-" * 100)

        if len(curr_upd) != len(page_inds):
            breakpoint()

        # map GPT text to PDF pages
        gptpagemap_section, textpagemap_section = map_gpttext_to_text(addedtext, curr_upd, page_inds, matcher)
        smry_fakepagemap_section = [-1] * len(sent_tokenize(smry))

        gpttext_all += addedtext + smry

        gptpagemap += gptpagemap_section + smry_fakepagemap_section
        textpagemap += textpagemap_section + [-1]

        logging.info(f"GPT paraphrazer output: \n\n{gpttext}\n")
        logging.info(f'Num of input tokens: {num_input_tokens}')
        logging.info(f'Num of output tokens: {num_output_tokens}')
        logging.info("-" * 100)
        logging.info(f'Original text pages: \n {page_inds} \n GPT text pages: \n {gptpagemap_section} \n')
        logging.info("-" * 100)

        verbalizer_steps.append([' '.join(curr_upd), gpttext])

        curr = ''
        page_inds = []
        curr_upd = []

        if len(sent_tokenize(gpttext_all)) != len(gptpagemap):
            raise Exception("Something went wrong. Mismatch between map and text")

    return gpttext_all, gptpagemap, verbalizer_steps, textpagemap


def map_gpttext_to_text(gpttext, text_splits, pagemap, matcher):
    # text_splits = sent_tokenize(text)

    blocks = []
    for p in pagemap:
        blocks.append((p[1], p[2]))

    text_blocks = []
    text_blocks_map = []
    for b in sorted(list(set(blocks))):
        block = []
        m = -1
        for i, p in enumerate(pagemap):
            if p[1] == b[0] and p[2] == b[1]:
                block.append(text_splits[i])
                m = p
        text_blocks.append(' '.join(block))
        text_blocks_map.append(m)

    gpttext_splits = sent_tokenize(gpttext)
    # text_splits = sent_tokenize(text)

    seq = matcher.match(gpttext_splits, text_splits, bert=True, fuzz=True, spacy=True, diff=True, tfidf=True, pnt=True)

    seq[0] = 0
    seq[-1] = len(text_splits)-1
    smoothed_seq = smooth_sequence(seq)

    gpt_blocks_map = [pagemap[s] for s in smoothed_seq]

    return gpt_blocks_map, text_blocks_map


def smooth_sequence(L1):
    N = len(L1)
    M = max(L1)

    # Initialize memo table with infinities
    memo = [[float('inf') for _ in range(M + 1)] for _ in range(N)]

    # Base case
    for j in range(M + 1):
        memo[0][j] = abs(L1[0] - j)

    # DP recurrence
    for i in range(1, N):
        for j in range(M + 1):
            for k in range(j + 1):
                current_loss = abs(L1[i] - j)
                memo[i][j] = min(memo[i][j], memo[i - 1][k] + current_loss)

    # Reconstruct the optimal sequence
    L2 = [0] * N
    last_val = memo[N - 1].index(min(memo[N - 1]))
    L2[N - 1] = last_val
    for i in range(N - 2, -1, -1):
        min_loss = float('inf')
        for j in range(last_val + 1):
            if memo[i][j] + abs(L2[i + 1] - last_val) < min_loss:
                min_loss = memo[i][j] + abs(L2[i + 1] - last_val)
                L2[i] = j
        last_val = L2[i]

    # Compute the total loss
    total_loss = sum([abs(L2[i] - L1[i]) for i in range(N)])

    return L2


def gpt_text_verbalizer(text, llm_api, llm_base, manual, include_summary, logging):

    encoding = tiktoken.get_encoding("cl100k_base")

    # split text into sections
    sections = text.split(' Section: ')

    curr = ''
    gpttext_all = ''
    verbalizer_steps = []

    for i_s, s in enumerate(sections):

        if len(s.split(' ###.')) < 2 or len(s.split(' ###.')[1]) == 0:
            continue

        curr += s
        if len(encoding.encode(curr)) < 5000:
            continue

        result = re.search(r'###(.*?)###', curr)
        if result:
            sectionName = result.group(1)
        else:
            sectionName = ''

        curr_upd = curr.replace("###", '')

        sys_message_func = "You are an ArXiv paper audio paraphraser. Your primary goal is to " \
                           "rephrase the original paper content while preserving its overall " \
                           "meaning and structure, but simplifying along the way, and make it easier to understand. " \
                           "In the event that you encounter a " \
                           "mathematical expression, it is essential that you verbalize it in " \
                           "straightforward nonlatex terms, while remaining accurate, and " \
                           "in order to ensure that the reader can grasp the equation's " \
                           "meaning solely through your verbalization. Do not output any long " \
                           "latex expressions, summarize them in words."

        human_message = 'The text must be written in the first person plural point of view. Do not use long latex ' \
                        'expressions, paraphrase them or summarize in words. Be as faithful to the given text ' \
                        'as possible. But ensure the paraphrasing is engaging and educational. Below is the ' \
                        'section of the paper, requiring paraphrasing and simplification and it is indicated by ' \
                        f'double angle brackets <<{curr_upd}>>. Do not output any of your own pre or post generation ' \
                        'statements. Only the resulting paraphrased text.'

        messages = [
            {'role': 'system', 'content': sys_message_func},
            {'role': 'user', 'content': human_message}
        ]

        logging.info(f'Section: {sectionName}\n')
        logging.info(sys_message_func + ' ' + human_message)
        logging.info("-" * 100)

        if manual:
            pyperclip.copy(sys_message_func + ' ' + human_message)
            val = input('\nEnter GPT output:')
            if val == '-1':
                break
            gpttext = pyperclip.paste()
            num_input_tokens = len(encoding.encode(sys_message_func + ' ' + human_message))
            num_output_tokens = len(encoding.encode(gpttext))
        else:
            for i in range(3):
                try:
                    response = llm_api(model=llm_base, messages=messages, temperature=0)
                    gpttext = response.choices[0].message.content
                    num_input_tokens = response.usage.prompt_tokens
                    num_output_tokens = response.usage.completion_tokens
                    break
                except:
                    time.sleep(5)
            else:
                raise Exception(f"{llm_base} failed")

        gpttext = gpttext.replace("$", '').replace("```", '').replace("<<", '').replace(">>", '').replace("**", '')
        # remove words with underscores
        gpttext = re.sub(r'\b\w*__\w*\b', '', gpttext)

        # replace double newlines with newline
        gpttext = re.sub('\n+', '\n', gpttext)

        addedtext = f' Section: {sectionName}. ' + gpttext + ' '

        smry = ''
        if include_summary:
            sys_message = "As an AI specializing in summarizing ArXiv paper sections, your main task is to distill "\
                          "complex scientific concepts from a given section of a paper into 2-3 simple, yet " \
                          "substantial, sentences. Retain key information, deliver the core idea, and ensure " \
                          "the summary is easy to understand, while not losing the main essence of the content. " \
                          "The text must be written in the first person plural point of view."

            human_message = 'Below is the section that needs to be summarized in at most 2-3 sentences and it is '\
                            f'indicated by double angle brackets <<{curr_upd}>>.'

            messages = [
                {'role': 'system', 'content': sys_message},
                {'role': 'user', 'content': human_message}
            ]

            for i in range(3):
                try:
                    response = llm_api(model=llm_base, messages=messages, temperature=0)
                    summary = response.choices[0].message.content
                    break
                except:
                    time.sleep(5)
            else:
                raise Exception(f"{llm_base} failed")

            smry = ' Section Summary: ' + summary + ' '

            logging.info(f"GPT summary: \n\n{summary}\n")
            logging.info("-" * 100)

        gpttext_all += addedtext + smry

        logging.info(f"GPT paraphrazer output: \n\n{gpttext}\n")
        logging.info(f'Num of input tokens: {num_input_tokens}')
        logging.info(f'Num of output tokens: {num_output_tokens}')
        logging.info("-" * 100)

        curr = ''

        verbalizer_steps.append([curr_upd, gpttext])

    return gpttext_all, verbalizer_steps


# Summarize Abstract
def create_summary(abstract, title, paper_id, llm_api, llm_base, files_dir):
    sys_message = "Given the abstract for the paper, summarize it in 30 words or less. " \
                  "Make sure the generated summary contains no more than 30 words."
    human_message = abstract
    messages = [
        {'role': 'system', 'content': sys_message},
        {'role': 'user', 'content': human_message}
    ]
    for i in range(3):
        try:
            response = llm_api(model=llm_base, messages=messages, temperature=0)
            summary = response.choices[0].message.content
            break
        except:
            time.sleep(5)
    else:
        summary = ''

    with open(os.path.join(files_dir, 'abstract.txt'), 'w') as f:
        f.write(title)
        f.write('\n\n')
        f.write(summary)
        f.write('\n\n')
        f.write(f'https://arxiv.org/abs//{paper_id}\n\n')
        f.write('YouTube: https://www.youtube.com/@ArxivPapers\n\nTikTok: https://www.tiktok.com/@arxiv_papers\n\nApple Podcasts: https://podcasts.apple.com/us/podcast/arxiv-papers/id1692476016\n\nSpotify: https://podcasters.spotify.com/pod/show/arxiv-papers\n')

