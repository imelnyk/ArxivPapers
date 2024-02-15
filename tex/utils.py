import os
import shutil
import tarfile
import wget
from glob import glob

def remove_oldfiles_samepaper(paper_id):
    if os.path.exists(paper_id):
        os.remove(paper_id)

    files_path = f'{paper_id}_files'
    if os.path.exists(files_path):
        shutil.rmtree(files_path)


def download_paper(paper_id):
    file = wget.download(f"https://arxiv.org/e-print/{paper_id}")
    files_dir = f'{paper_id}_files'
    os.makedirs(files_dir, exist_ok=True)

    with tarfile.open(file) as tar:
        tar.extractall(files_dir)

    return files_dir


def get_tex_files(files_dir):
    tex_files = glob(os.path.join(files_dir, '*.tex'))
    return [os.path.splitext(os.path.basename(c))[0] for c in tex_files]


def get_main_source_file(tex_files, files_dir):
    for file in tex_files:
        with open(os.path.join(files_dir, f'{file}.tex')) as f:
            content = f.read()

        # Check for documentclass and begin/end document tags
        if "\\documentclass" in content and "\\begin{document}" in content and "\\end{document}" in content:
            return file

    return None

