import os
import shutil
import tarfile
import wget
from glob import glob
import subprocess


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
        if isinstance(points[0], list):
            points = points[0]

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

def create_questions(questions, dir_path):

    def generate_beamer_slide(question):
        slide = \
        '''
        \\documentclass[20pt]{beamer}\n
        \\usetheme{default}\n
        \\geometry{papersize={8.5in,11in}}
        \\setbeamertemplate{navigation symbols}{}\n
        \\setbeamertemplate{footline}{}\n
        \\setbeamertemplate{headline}{}\n        
        \\usepackage{tikz}\n
        \\begin{document}\n
        \\begin{frame}[plain]\n
            \\vfill\n
            \\begin{center}\n
                \\begin{tikzpicture}\n
        '''
        slide += "\\node[draw, rounded corners=5pt, fill=blue!20, minimum width=0.95\\textwidth, " \
                 "minimum height=2cm, text centered, text width=0.95\\textwidth, inner sep=10pt] (Question) { "\
                 "\\textbf{\\large %s}};" % question

        slide += \
        '''
                \\end{tikzpicture}\n
            \\end{center}\n
            \\vfill\n
        \\end{frame}\n
        \\end{document}\n
        '''

        return slide

    def compile_latex_to_pdf(latex_code, directory, filename="presentation"):

        # Save the LaTeX code to a .tex file
        with open(os.path.join(directory, f"{filename}.tex"), 'w') as f:
            f.write(latex_code)

        # Compile using pdflatex
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{filename}.tex"], cwd=directory)

        print(f"{filename}.pdf generated!")


    os.makedirs(dir_path, exist_ok=True)

    for i, Q in enumerate(questions):
        slide_code = generate_beamer_slide(Q)
        compile_latex_to_pdf(slide_code, directory=dir_path, filename=f"question_{i}")

    # List all files in the directory
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    # List all .pdf files in the directory
    pdf_files = [os.path.basename(f) for f in glob(os.path.join(dir_path, "*.pdf"))]

    # Subtract the list of pdf_files from all_files to get files to delete
    files_to_delete = set(all_files) - set(pdf_files)

    # Iterate over the files to delete and remove them
    for file in files_to_delete:
        os.remove(os.path.join(dir_path, file))