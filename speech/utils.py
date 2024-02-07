from nltk.tokenize import sent_tokenize
from google.cloud import texttospeech
import os
import textwrap
import subprocess
from glob import glob

def text_to_speech_short(text, slides, mp3_list_file, files_dir, tts_client, logging):
    para = text.split('\n\n')
    slides = slides['slides']

    if len(para) == len(slides):
        first_para_sent = sent_tokenize(para[0])
        para[0] = " ".join(first_para_sent[1:])
        para.insert(0, first_para_sent[0])

    for s in para:
        response = synthesize_speech(s, tts_client, 'Neural2-F', 1.0)

        logging.info(f'Processed block text: \n\n {s}')
        logging.info("-" * 100)

        chunk_audio_file_name = f'short_{hash(s)}.mp3'
        chunk_audio = os.path.join(files_dir, f'{chunk_audio_file_name}')

        with open(chunk_audio, "wb") as out:
            out.write(response.audio_content)

        if os.path.getsize(chunk_audio) == 0:
            continue

        mp3_list_file.write(f'file {chunk_audio_file_name}\n')


def synthesize_speech(text, tts_client, voice, rate=1.0):
    voice_params = texttospeech.VoiceSelectionParams(language_code="en-US", name=f"en-US-{voice}")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=rate)

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return response
    except Exception as e:
        return None


def text_to_speechvideo(text, mp3_list_file, files_dir, tts_client, pageblockmap, voice, logging):
    splits = sent_tokenize(text)

    assert len(pageblockmap) == len(splits), "Number of pageblockmap does not match number of splits"

    block_text = []
    prev = pageblockmap[0]
    last_page = 0

    for ind, m in enumerate(pageblockmap):
        if m == prev:
            block_text.append(splits[ind])
            continue

        joinedtext = ' '.join(block_text)
        if isinstance(prev, list):
            last_page = prev[1]
            synthesize(joinedtext, tts_client, files_dir, mp3_list_file,
                       page=prev[1], block=prev[2], voice=voice, logging=logging)
        else:
            synthesize(joinedtext, tts_client, files_dir, mp3_list_file,
                       page=last_page, voice=voice, logging=logging)

        prev = m
        block_text = [splits[ind]]


def synthesize(text, tts_client, files_dir, mp3_list_file, page=None, block=None, voice=None, logging=None):
    response = synthesize_speech(text, tts_client, voice)

    # if processing fails, subdivide into smaller chunks and try again
    if response:
        logging.info(f'Processed block text: \n\n {text}')
        logging.info("-" * 100)

        if block is None:
            chunk_audio_file_name = f'page{page}summary_{hash(text)}.mp3'
        else:
            chunk_audio_file_name = f'page{page}block{block}_{hash(text)}.mp3'

        chunk_audio = os.path.join(files_dir, f'{chunk_audio_file_name}')
        with open(chunk_audio, "wb") as out:
            out.write(response.audio_content)

        mp3_list_file.write(f'file {chunk_audio_file_name}\n')
    else:
        chunks = textwrap.wrap(text,
                               width=len(text) // 2,
                               break_long_words=False,
                               expand_tabs=False,
                               replace_whitespace=False,
                               drop_whitespace=False,
                               break_on_hyphens=False)

        synthesize(chunks[0], tts_client, files_dir, mp3_list_file, page, block, voice, logging)
        synthesize(chunks[1], tts_client, files_dir, mp3_list_file, page, block, voice, logging)


def text_to_speech(text, mp3_list_file, files_dir, tts_client, voice, logging):
    splits = sent_tokenize(text)

    block = []

    for s in splits:

        block.append(s)
        if len(block) < 3:
            continue

        block_text = ' '.join(block)
        response = synthesize_speech(block_text, tts_client, voice)

        logging.info(f'Processed text: \n\n {block_text}')
        logging.info("-" * 100)

        chunk_audio_file_name = f'part_{hash(block_text)}.mp3'
        chunk_audio = os.path.join(files_dir, f'{chunk_audio_file_name}')

        with open(chunk_audio, "wb") as out:
            out.write(response.audio_content)

        mp3_list_file.write(f'file {chunk_audio_file_name}\n')
        block = []


def create_slides(slides, dir_path):
    def generate_beamer_slide(title, bullet_points):
        """
        Generate a LaTeX Beamer slide with given title and bullet points.

        :param title: Title of the slide
        :param bullet_points: List of bullet points. If a bullet point is a tuple,
                              the first element is the main point, and the second is a list of sub-points.
        :return: LaTeX code as a string.
        """
        slide = "\\begin{frame}\n"
        slide += "  \\frametitle{%s}\n" % title
        slide += "  \\begin{itemize}\n"

        for point in bullet_points:
            if isinstance(point, tuple):  # main point with sub-points
                slide += "    \\item %s\n" % point[0]
                slide += "    \\begin{itemize}\n"
                for sub_point in point[1]:
                    slide += "      \\item %s\n" % sub_point
                slide += "    \\end{itemize}\n"
            else:  # just a main point
                slide += "    \\item %s\n" % point

        slide += "  \\end{itemize}\n"
        slide += "\\end{frame}\n"

        return slide

    def compile_latex_to_pdf(latex_code, directory, filename="presentation"):
        # Complete the LaTeX code with preamble and custom dimensions
        full_code = f"""
        \\documentclass[20pt]{{beamer}}
        \\geometry{{papersize={{8.5in,11in}}}}
        \\begin{{document}}
        {latex_code}
        \\end{{document}}
        """

        latex_filename = f"{filename}.tex"

        # Save the LaTeX code to a .tex file
        with open(os.path.join(directory, latex_filename), 'w') as f:
            f.write(full_code)

        # Compile using pdflatex
        subprocess.run(["pdflatex",  "-interaction=nonstopmode", f"{filename}.tex"], cwd=directory)

        print(f"{filename}.pdf generated!")

    os.makedirs(dir_path, exist_ok=True)

    for i, slide in enumerate(slides['slides']):
        title = slide[0]
        points = slide[1:]

        slide_code = generate_beamer_slide(title, points)
        compile_latex_to_pdf(slide_code, directory=dir_path, filename=f"slide_{i+1}")

    # List all files in the directory
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    # List all .pdf files in the directory
    pdf_files = [os.path.basename(f) for f in glob(os.path.join(dir_path, "*.pdf"))]

    # Subtract the list of pdf_files from all_files to get files to delete
    files_to_delete = set(all_files) - set(pdf_files)

    # Iterate over the files to delete and remove them
    for file in files_to_delete:
        os.remove(os.path.join(dir_path, file))
