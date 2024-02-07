import logging
import os
import bs4
from bs4 import BeautifulSoup
import re


def generate_html(main_file, files_dir, html_parser, pdflatex, latex2html, latexmlc, gs, display, logging):
    prev_dir = os.getcwd()
    os.chdir(files_dir)

    os.system(f'{pdflatex} -interaction=nonstopmode {main_file}.tex {display}')
    os.system(f'{pdflatex} -interaction=nonstopmode {main_file}.tex {display}')

    if html_parser == 'latex2html':
        logging.info(f'Started processing with latex2html...')
        os.system(f'{latex2html} {main_file}.tex {display}')
    else:
        logging.info(f'Started processing with latexmlc...')
        os.system(f'{latexmlc} --timeout=6000 --navigationtoc=context --mathtex --dest=./{main_file}/{main_file}.html {main_file}.tex {display}')

    logging.info(f'Done.')

    # make sure it is 8.5in x 11in
    os.system(f'{gs} -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/default -dNOPAUSE -dQUIET -dBATCH -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=612 -dDEVICEHEIGHTPOINTS=792 -sOutputFile=tmp.pdf {main_file}.pdf {display}')
    os.replace('tmp.pdf', f'{main_file}.pdf')

    os.chdir(prev_dir)


def parse_aux_file(aux_file_path):
    with open(aux_file_path, 'r', errors='ignore') as aux_file:
        aux_lines = aux_file.readlines()
    return [line[line.find('{') + 1:line.find('}')] for line in aux_lines if '\\bibcite' in line]


def parse_html_file(html_file_path, html_parser):
    with open(html_file_path) as html_file:
        soup = BeautifulSoup(html_file, "lxml")

    title = soup.head.title.get_text().replace("\n", " ")

    if html_parser == 'latex2html':
        content = soup.find(class_='ChildLinks')
        if soup.find(class_='ABSTRACT') is None:
            abstract = ' '
        else:
            abstract = soup.find(class_='ABSTRACT').get_text()
    else:
        content = soup.find(class_='ltx_toclist')
        abstract = soup.find(class_='ltx_abstract').get_text()

    return title, content, soup, abstract

def create_nested_dict(ul_element, logging, stop_words):
    nested_dict = {}

    for li in ul_element.find_all("li", recursive=False):

        if li.has_attr('class'):
            for c in li['class']:
                if 'appendix' in c:
                    return nested_dict

        anchor_tag = li.find('a')
        title = anchor_tag.get_text()
        # stop_words = []
        for w in stop_words:
            if w in title.lower():
                return nested_dict

        logging.info(f'{title = }')

        link = anchor_tag['href']
        sub_list_element = li.find("ul") or li.find("ol")

        if sub_list_element:
            nested_dict[title] = [{link: ''}, create_nested_dict(sub_list_element, logging, stop_words)]
        else:
            nested_dict[title] = [{link: ''}, None]

    return nested_dict


def extract_text_recursive(D, dir_loc, main_file, citations, html_parser, html):
    for sec in list(D.keys()):
        href = next(iter(D[sec][0]))
        local_text = []
        if html_parser == 'latex2html':
            html_file = os.path.join(dir_loc, main_file, f"{href}")
            with open(html_file, encoding='utf-8', errors='ignore') as f:
                H = BeautifulSoup(f, "lxml")
                parse_text(local_text, H.body, html_parser)
        else:
            section = html.find(id=href[1:])
            if section is not None:
                parse_text(local_text, section, html_parser)

        if local_text:
            processed_text = clean_text(''.join(local_text), citations)
            D[sec][0][href] = processed_text

        if D[sec][1]:
            extract_text_recursive(D[sec][1], dir_loc, main_file, citations, html_parser, html)


def clean_text(text, citations):
    for cite in citations:
        text = text.replace(cite, '')

    delete_items = ['=-1', '\t', '\n', u'\xa0', '[]', '()', '\\', 'mathbb', 'mathcal', 'bm', 'mathrm', 'mathit',
                    'mathbf', 'mathbfcal', 'textbf', 'textsc', 'langle', 'rangle', 'mathbin']

    for item in delete_items:
        text = text.replace(item, '')

    text = re.sub(' +', ' ', text)
    text = re.sub(r'[[,]+]', '', text)

    #ensure that every period has a space after it (excluding digits)
    text = re.sub(r'\.(?!\d)', '. ', text)

    return text


def parse_text(local_text, tag, html_parser):
    ignore_tags = ['a', 'figure', 'center', 'caption', 'td', 'h1', 'h2', 'h3', 'h4']
    if html_parser == 'latex2html':
        ignore_tags += ['table'] # 'div'
    else:
        ignore_tags += ['sup', 'cite']
    max_math_length = 300000

    for child in tag.children:
        child_type = type(child)

        if child_type == bs4.element.NavigableString:
                txt = child.get_text()
                local_text.append(txt)

        elif child_type == bs4.element.Comment:
            continue
        elif child_type == bs4.element.Tag:

                if child.name in ignore_tags or (child.has_attr('class') and child['class'][0] == 'navigation'):
                    continue
                elif child.name == 'img' and child.has_attr('alt'):
                    # math_txt = child.get('alt').replace("$", "")
                    math_txt = child.get('alt')
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.has_attr('class') and (child['class'][0] == 'ltx_Math' or child['class'][0] == 'ltx_equation'):
                    # math_txt = child.get_text().replace("\\", "")
                    math_txt = child.get_text()
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.name == 'section':
                    return
                else:
                    parse_text(local_text, child, html_parser)
        else:
            raise RuntimeError('Unhandled type')


def depth_first_search(document, current_document=None, joint_text='', visited=None):
    if visited is None:
        visited = set()

    if current_document is None:
        current_document = document

    for section, (text_obj, subsections) in current_document.items():
        text = list(text_obj.values())[0]
        text_hash = hash(text)

        section = section.rstrip()
        if section.endswith('.'):
            section = section[:-1]

        if text_hash not in visited:
            joint_text += f' Section: ### {section}. ###. {text}'
            visited.add(text_hash)

        if subsections:
            joint_text = depth_first_search(document, subsections, joint_text, visited)

    return joint_text
