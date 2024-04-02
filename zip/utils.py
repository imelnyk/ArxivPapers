import os
from PyPDF2 import PdfReader, PdfWriter
from glob import glob
import datetime
import zipfile


def crop_pdf(input_pdf_path, output_pdf_path, gs, upper_top, top_percent, left_percent, right_percent):

    with open(input_pdf_path, "rb") as file_handle:
        pdf = PdfReader(file_handle)
        pdf_writer = PdfWriter()

        page = pdf.pages[0]

        # MediaBox defines the boundaries of the physical medium
        mediabox = page.mediabox

        # Calculation
        to_crop_top = mediabox[3] * top_percent / 100
        to_crop_right = mediabox[2] * right_percent / 100
        to_crop_left = mediabox[2] * left_percent / 100
        to_crop_upper_top = mediabox[3] * upper_top / 100

        # Adjusting the media box values
        page.mediabox.upper_right = (float(mediabox[2]) - float(to_crop_right), float(mediabox[3]) - float(to_crop_upper_top))
        page.mediabox.upper_left = (float(to_crop_left), float(mediabox[3]) - float(to_crop_upper_top))
        page.mediabox.lower_left = (float(to_crop_left), float(mediabox[3]) - float(to_crop_top))

        page.mediabox.lower_right = (float(mediabox[2]) - float(to_crop_right), float(mediabox[3]) - float(to_crop_top))

        pdf_writer.add_page(page)

        with open(output_pdf_path, "wb") as out:
            pdf_writer.write(out)

    os.system(f'{gs} -sDEVICE=png16m -r500 -o {output_pdf_path}.png {output_pdf_path}')


def zip_files(files_dir, gs, ffmpeg, create_short, create_qa, create_video, final_audio_file, chunk_mp3_file_list, display):

    files_to_zip = []
    if create_short:
        files_to_zip = [f'{final_audio_file}_short.mp3', f'shorts_{chunk_mp3_file_list}']

        if os.path.exists(os.path.join(files_dir, 'slides')):

            abs_paths = glob(os.path.join(files_dir, 'slides', 'slide*.pdf'))
            relative_paths = [os.path.relpath(p, files_dir) for p in abs_paths]
            files_to_zip.extend(relative_paths)

            with open(os.path.join(files_dir, f'shorts_{chunk_mp3_file_list}'), "r") as f:
                lines = f.readlines()

            # Process each line
            for line in lines:
                # Remove the newline character at the end of the line
                line = line.strip()

                # Split the line into components
                components = line.split()

                # The filename is the second component
                audio = components[1]

                files_to_zip.append(audio)

        # short video only
        if not create_video:
            page_num = 0
            audio = 'final_audio_short'
            resolution = "scale=1920:-2"
            video = 'output_short'
            # extract first page of PDF
            os.system(f'{gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={page_num + 1} '
                      f'-dLastPage={page_num + 1} -sOutputFile={os.path.join(files_dir, str(page_num))}.pdf '
                      f'{os.path.join(files_dir, "main.pdf")} {display}')
            # convert to PNG
            os.system(f'{gs} -sDEVICE=png16m -r300 -o {os.path.join(files_dir, str(page_num))}.png '
                      f'{os.path.join(files_dir, str(page_num))}.pdf {display}')
            os.system(f'{ffmpeg} -loop 1 -i {os.path.join(files_dir, str(page_num))}.png '
                      f'-i {os.path.join(files_dir, audio)}.mp3 -vf {resolution} -c:v libx264 -tune stillimage '
                      f'-y -c:a aac -b:a 128k -pix_fmt yuv420p -shortest {os.path.join(files_dir, video)}.mp4 {display}')

    if create_qa:
        files_to_zip.extend([f'{final_audio_file}_qa.mp3', f'qa_{chunk_mp3_file_list}', 'qa_pages.pkl'])

        if os.path.exists(os.path.join(files_dir, 'questions')):

            abs_paths = glob(os.path.join(files_dir, 'questions', 'question*.pdf'))
            relative_paths = [os.path.relpath(p, files_dir) for p in abs_paths]
            files_to_zip.extend(relative_paths)

            with open(os.path.join(files_dir, f'qa_{chunk_mp3_file_list}'), "r") as f:
                lines = f.readlines()

            # Process each line
            for line in lines:
                # Remove the newline character at the end of the line
                line = line.strip()

                # Split the line into components
                components = line.split()

                # The filename is the second component
                audio = components[1]

                files_to_zip.append(audio)

    if create_video:
        with open(f"{os.path.join(files_dir, chunk_mp3_file_list)}", "r") as f:
            lines = f.readlines()

        files_to_zip.extend(['main.pdf', chunk_mp3_file_list, 'block_coords.pkl', 'gptpagemap.pkl'])

        # Process each line
        for line in lines:
            # Remove the newline character at the end of the line
            line = line.strip()

            # Split the line into components
            components = line.split()

            # The filename is the second component
            audio = components[1]

            files_to_zip.append(audio)

        original_directory = os.getcwd()
        os.chdir(files_dir)

        # time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # Outputs as '20230709_174500'
        # zip_file_name = f'zipfile-{time_str}.zip'
        zip_file_name = f'zipfile.zip'
        with zipfile.ZipFile(zip_file_name, 'w') as zipf:
            # Loop through the list of files
            for file in files_to_zip:
                # Add each file to the zip file
                zipf.write(file)

        os.chdir(original_directory)