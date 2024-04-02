import pickle
import argparse
import openai
from tex.utils import *
from htmls.utils import *
from map.utils import *
from gpt.utils import *
from speech.utils import *
from zip.utils import *
from gdrive.utils import *
import random


def main(args):

    if args.verbose == 'debug':
        display = ''
        level = logging.INFO
    elif args.verbose == 'info':
        display = '> /dev/null 2>&1'
        level = logging.INFO
    else:
        display = '> /dev/null 2>&1'
        level = logging.WARNING

    logging.basicConfig(level=level, format='\n%(asctime)s - %(levelname)s - %(message)s')

    # ==============================================================================================================
    #                                                   LATEX
    # ==============================================================================================================
    paper_id = args.paperid

    remove_oldfiles_samepaper(paper_id)
    files_dir = download_paper(paper_id)

    tex_files = get_tex_files(files_dir)
    logging.info(f'Found .tex files: {tex_files}')

    main_file = get_main_source_file(tex_files, files_dir)

    if main_file:
        logging.info(f'Found [{main_file}.tex] as a main source file')
    else:
        logging.info(f'Failed to find main source file. Enter manually:')
        main_file = input()

    # ==============================================================================================================
    #                                                   HTML
    # ==============================================================================================================

    html_parser = 'latex2html' if args.l2h else 'latexmlc'

    generate_html(main_file, files_dir, html_parser, args.pdflatex,
                  args.latex2html, args.latexmlc, args.gs, display, logging)

    citations = parse_aux_file(os.path.join(files_dir, f'{main_file}.aux'))

    title, content, html, abstract = parse_html_file(os.path.join(files_dir, main_file, f'{main_file}.html'), html_parser)

    logging.info(f'Parsed HTML file. Title: {title}')

    stop_words = ['references', 'appendix', 'conclusion', 'acknowledgments', 'about this document']
    if args.stop_word:
        stop_words.append(args.stop_word)

    logging.info(f'Section title stop words: {stop_words}')

    D = create_nested_dict(content, logging, stop_words)

    extract_text_recursive(D, files_dir, main_file, citations, html_parser, html)

    text = depth_first_search(D)

    # to make sure splitting is repeatable
    splits = sent_tokenize(text)
    text = ' '.join(splits)

    with open(os.path.join(files_dir, 'extracted_orig_text_clean.txt'), 'w') as f:
        f.write(text)

    # dump text split into sections
    with open(os.path.join(files_dir, 'original_text_split_sections.txt'), 'w') as f:
        sections = text.split(' Section: ')
        for i, s in enumerate(sections):
            f.write(f'SECTION {i + 1}\n\n')
            f.write(s)
            f.write("\n\n")

    if args.extract_text_only:
        return

    # ==============================================================================================================
    #                                                   MAP
    # ==============================================================================================================

    if args.create_video:
        matcher = Matcher(args.cache_dir)

        logging.info('Mapping text to pages')
        pagemap = map_text_to_pdfpages(text, f'{os.path.join(files_dir, main_file)}.pdf', matcher)

        logging.info('Mapping pages to blocks')
        coords, pageblockmap = map_page_to_blocks(pagemap, text, args.gs, files_dir,
                                          f'{os.path.join(files_dir, main_file)}.pdf', matcher, display)

        with open(os.path.join(files_dir, 'block_coords.pkl'), 'wb') as f:
            pickle.dump(coords, f)

        # dump text split into pages
        with open(os.path.join(files_dir, 'original_text_split_pages.txt'), 'w') as f:
            splits = sent_tokenize(text)

            for i, p in enumerate(np.unique(pagemap)):
                start = np.where(np.array(pagemap) == p)[0][0]
                end = np.where(np.array(pagemap) == p)[0][-1] + 1
                chunk_text = ' '.join(splits[start:end])
                f.write(f'PAGE {p + 1}\n\n')
                f.write(chunk_text)
                f.write("\n\n")

    # ==============================================================================================================
    #                                                   GPT
    # ==============================================================================================================

    openai.api_key = args.openai_key
    llm_api = openai.chat.completions.create

    logging.info('GPT Paraphrasing')
    tmpdata = {}
    if args.create_short:
        gpttext_short, slides_short = gpt_short_verbalizer(files_dir, llm_api, args.llm_strong, args.llm_base, logging)
        with open(os.path.join(files_dir, 'gpt_text_short.txt'), 'w') as f:
            f.write(gpttext_short)

        with open('gpt_slides_short.json', 'w') as json_file:
            json.dump(slides_short, json_file, indent=4)

        tmpdata = {'gpttext_short': gpttext_short, 'gptslides_short': slides_short['slides']}

    if args.create_qa:
        questions, answers, qa_pages = gpt_qa_verbalizer(files_dir, llm_api, args.llm_base, matcher, logging)

        create_questions(questions, os.path.join(files_dir, 'questions'))

        with open(os.path.join(files_dir, 'qa_pages.pkl'), 'wb') as f:
            pickle.dump(qa_pages, f)

        with open(os.path.join(files_dir, 'gpt_questions_answers.txt'), 'w') as f:
            for q, a in zip(questions, answers):
                f.write(f'==== Question ====\n\n')
                f.write(q)
                f.write("\n\n")
                f.write(f'==== Answer ====\n\n')
                f.write(a)
                f.write("\n\n")

        tmpdata = {'gpttext_q': questions, 'gpttext_a': answers, 'qa_pages': qa_pages}

    if args.create_video:
        (gpttext, gptpagemap,
         verbalizer_steps, textpagemap) = gpt_textvideo_verbalizer(text,
                                                                   llm_api,
                                                                   args.llm_strong,
                                                                   args.llm_base,
                                                                   args.manual_gpt,
                                                                   args.include_summary,
                                                                   pageblockmap,
                                                                   matcher,
                                                                   logging)

        with open(os.path.join(files_dir, 'gptpagemap.pkl'), 'wb') as f:
            pickle.dump(gptpagemap, f)

        tmpdata.update({'gpttext': gpttext,
                        'gptpagemap': gptpagemap,
                        'verbalizer_steps': verbalizer_steps,
                        'textpagemap': textpagemap})

        with open(os.path.join(files_dir, 'gpt_verb_steps.txt'), 'w') as f:
            for si, s in enumerate(verbalizer_steps):
                f.write(f'===Original {si}===\n\n')
                f.write(s[0])
                f.write("\n\n")
                f.write(f'===GPT {si}===\n\n')
                f.write(s[1])
                f.write("\n\n")

        with open(os.path.join(files_dir, 'gpt_text.txt'), 'w') as f:
            f.write(gpttext)

        logging.info(f'Extracted text:\n\n {gpttext}')

    if len(tmpdata) > 0:
        with open(os.path.join(files_dir, 'tmpdata.pkl'), 'wb') as f:
            pickle.dump(tmpdata, f)

    if args.create_audio_simple:
        gpttext, verbalizer_steps = gpt_text_verbalizer(text, llm_api, args.llm_base, args.manual_gpt, args.include_summary, logging)

    # ==============================================================================================================
    #                                                   SPEECH
    # ==============================================================================================================

    tts_client = texttospeech.TextToSpeechClient()
    if args.gdrive_id:
        gdrive_client = GDrive(args.gdrive_id)

    if args.create_short:
        with open(os.path.join(files_dir, args.chunk_mp3_file_list), 'w') as mp3_list_file:
            text_to_speech_short(gpttext_short, slides_short, mp3_list_file, files_dir, tts_client, logging)

        shutil.copy(os.path.join(files_dir, args.chunk_mp3_file_list),
                    os.path.join(files_dir, f'shorts_{args.chunk_mp3_file_list}'))

        final_audio_short = os.path.join(files_dir, f'{args.final_audio_file}_short.mp3')
        os.system(f'{args.ffmpeg} -f concat -i {os.path.join(files_dir, args.chunk_mp3_file_list)} '
                  f'-c copy {final_audio_short} {display}')

        logging.info(f'Created short audio file')

        if args.gdrive_id:
            gdrive_client.upload_audio(f'[short] {title}', f'{final_audio_short}')
            logging.info(f'Uploaded short audio to GDrive')

        create_slides(slides_short, os.path.join(files_dir, 'slides'))

    if args.create_qa:
        with open(os.path.join(files_dir, args.chunk_mp3_file_list), 'w') as mp3_list_file:
            text_to_speech_qa(questions, answers, mp3_list_file, files_dir, tts_client, args.ffmpeg, logging)

        shutil.copy(os.path.join(files_dir, args.chunk_mp3_file_list),
                    os.path.join(files_dir, f'qa_{args.chunk_mp3_file_list}'))

        final_audio_qa = os.path.join(files_dir, f'{args.final_audio_file}_qa.mp3')
        os.system(f'{args.ffmpeg} -f concat -i {os.path.join(files_dir, args.chunk_mp3_file_list)} '
                  f'-c copy {final_audio_qa} {display}')

        logging.info(f'Created QA audio file')

        if args.gdrive_id:
            gdrive_client.upload_audio(f'[QA] {title}', f'{final_audio_qa}')
            logging.info(f'Uploaded QA audio to GDrive')

    if args.create_video:
        with open(os.path.join(files_dir, args.chunk_mp3_file_list), 'w') as mp3_list_file:
            text_to_speechvideo(gpttext, mp3_list_file, files_dir, tts_client, gptpagemap, args.voice, logging)

    if args.create_audio_simple:
        with open(os.path.join(files_dir, args.chunk_mp3_file_list), 'w') as mp3_list_file:
            text_to_speech(gpttext, mp3_list_file, files_dir, tts_client, args.voice, logging)

    final_audio = os.path.join(files_dir, f'{args.final_audio_file}.mp3')
    os.system(f'{args.ffmpeg} -f concat -i {os.path.join(files_dir, args.chunk_mp3_file_list)} -c copy {final_audio} {display}')
    logging.info(f'Created audio file')

    if args.gdrive_id:
        gdrive_client.upload_audio(title, f'{final_audio}')
        logging.info(f'Uploaded audio to GDrive')

    # --------------- SUMMARY -----------------
    create_summary(abstract, title, paper_id, llm_api, args.llm_base, files_dir)

    # ==============================================================================================================
    #                                                   ZIP
    # ==============================================================================================================

    renamed_main = 'main'
    if main_file != renamed_main:
        temp_filename = f"temp_{random.randint(1000, 9999)}"
        shutil.copy(f'{os.path.join(files_dir, main_file)}.pdf', f'{os.path.join(files_dir, temp_filename)}.pdf')
        shutil.copy(f'{os.path.join(files_dir, temp_filename)}.pdf',f'{os.path.join(files_dir, renamed_main)}.pdf')

    crop_pdf(f"{files_dir}/{main_file}.pdf",f"{files_dir}/fpage.pdf", args.gs,
             upper_top=3, top_percent=25, left_percent=12, right_percent=7)

    zip_files(files_dir, args.gs, args.ffmpeg,
              args.create_short, args.create_video, args.final_audio_file,
              args.chunk_mp3_file_list, display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument("--paperid", type=str, default='', help='ArXiv paper id, e.g., 1706.03762')
    parser.add_argument("--l2h", action="store_true", help="use latex2html instead of latexmlc to convert latex to HTML")
    parser.add_argument("--verbose", type=str, choices=['silent', 'info', 'debug'], default='info', help="print debugging information: debug-all, info-some")
    parser.add_argument("--pdflatex", type=str, default='pdflatex', help='pdflatex command')
    parser.add_argument("--latex2html", type=str, default='latex2html', help='latex2html command')
    parser.add_argument("--latexmlc", type=str, default='latexmlc', help='latexmlc command')
    parser.add_argument("--stop_word", type=str, default='', help='word/phrase in section to stop HTML parsing at')
    parser.add_argument("--ffmpeg", type=str, default='ffmpeg', help='ffmpeg command')
    parser.add_argument("--gs", type=str, default='gs', help='gs command')
    parser.add_argument("--cache_dir", type=str, default='cache', help='cache directory for BERT model')
    parser.add_argument("--gdrive_id", type=str, default="", help="ID of the Google folder where to save audio files")
    parser.add_argument("--voice", type=str, default='Polyglot-1', help="voice name for Google text-to-speech model")
    parser.add_argument("--final_audio_file", type=str, default='final_audio', help="final audio name")
    parser.add_argument("--chunk_mp3_file_list", type=str, default='mp3_list.txt', help="where to save chunk mp3 files")
    parser.add_argument("--manual_gpt", action="store_true", help="manual entry of GPT outputs")
    parser.add_argument("--include_summary", action="store_true", help="include summary after each section")
    parser.add_argument("--extract_text_only", action="store_true", help="extract only the text from paper and exit")
    parser.add_argument("--create_video", action="store_true", help="create long video")
    parser.add_argument("--create_short", action="store_true", help="create short video")
    parser.add_argument("--create_qa", action="store_true", help="create qa video")
    parser.add_argument("--create_audio_simple", action="store_true", help="create audio")
    parser.add_argument("--openai_key", type=str, default="", help='openai key to call GPT API')
    parser.add_argument("--llm_strong", type=str, default="gpt-4-0125-preview", help='llm model for complex tasks')
    parser.add_argument("--llm_base", type=str, default="gpt-3.5-turbo-0125", help='llm model for basic tasks')

    args = parser.parse_args()

    assert not (args.create_audio_simple and args.create_video), "Cannot create long video and simple audio at once"

    main(args)