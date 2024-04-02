import os
import zipfile
import re
import fitz
from PIL import Image, ImageDraw
import shutil
import glob
import pickle
import argparse


def main(args):

    files = glob.glob(os.path.join(f"{args.paperid}_files", "zipfile.zip"))
    if files:
        zip_name = max(files, key=os.path.getmtime)
        # time_str = zip_name.split('-')[1].strip('.zip')
        dr = os.path.join(f"{args.paperid}_files", "output")
    else:
        return

    if os.path.exists(dr):
        shutil.rmtree(dr)

    with zipfile.ZipFile(zip_name, 'r') as zipf:
        # Extract all the contents of the zip file
        zipf.extractall(dr)

    with open(os.path.join(dr, "mp3_list.txt"), "r") as f:
        lines = f.readlines()

    # create list of chunks
    outvideo = open(os.path.join(dr, 'mp4_list.txt'), 'w')

    block_coords = pickle.load(open(os.path.join(dr, 'block_coords.pkl'), 'rb'))
    gptpagemap = pickle.load(open(os.path.join(dr, 'gptpagemap.pkl'), 'rb'))

    # Process each line
    for line in lines:

        # Remove the newline character at the end of the line
        line = line.strip()

        # Split the line into components
        components = line.split()

        # The filename is the second component
        audio = components[1].replace('.mp3', '')
        video = audio.replace('-', '')

        # The number is the fourth component (without the #)
        match = re.search(r'page(\d+)', components[1])
        page_num = int(match.group(1))

        # extract first page of PDF
        os.system(f'{args.gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={page_num+1} -dLastPage={page_num+1} -sOutputFile={os.path.join(dr, str(page_num))}.pdf {os.path.join(dr,"main.pdf")} > /dev/null 2>&1')

        # convert to PNG
        os.system(f'{args.gs} -sDEVICE=png16m -r300 -o {os.path.join(dr, str(page_num))}.png {os.path.join(dr, str(page_num))}.pdf')

        if 'summary' not in components[1]:

            doc = fitz.open(f'{os.path.join(dr, str(page_num))}.pdf')
            page = doc[0]

            match = re.search(r'block(\d+)', components[1])
            block_num = int(match.group(1))

            for pb in gptpagemap:
                if isinstance(pb, list):
                    if pb[1] == page_num and pb[2] == block_num:
                        coords = block_coords[pb[0]][block_num]
                        break

            # Load an image
            image = Image.open(f'{os.path.join(dr, str(page_num))}.png')

            # Calculate scale factors
            scale_x = image.width / page.rect.width
            scale_y = image.height / page.rect.height

            # Rescale coordinates
            x0, y0, x1, y1 = coords
            x0 *= scale_x
            y0 *= scale_y
            x1 *= scale_x
            y1 *= scale_y

            # find out if this rectangle is on the left, right or whole width
            if coords[2] > page.rect.width / 2:
                pointerleft = Image.open('imgs/pointertoleft.png')
                pointer = pointerleft.convert("RGBA")
                onrightside = True
            else:
                pointerright = Image.open('imgs/pointertoright.png')
                pointer = pointerright.convert("RGBA")
                onrightside = False

            # Create draw object
            draw = ImageDraw.Draw(image)

            # Define thickness
            thickness = 5

            # Draw several rectangles to simulate thickness
            for i in range(thickness):
                draw.rectangle([x0 - i - 5, y0 - i - 5, x1 + i + 5, y1 + i + 5], outline="green")

            # Calculate the center of the rectangle
            rect_center_y = (y0 + y1) / 2

            # Scale down the pointer image while preserving aspect ratio
            desired_height = image.height / 20
            aspect_ratio = pointer.width / pointer.height
            new_width = int(aspect_ratio * desired_height)
            pointer = pointer.resize((new_width, int(desired_height)))

            # Calculate position for the pointer
            if onrightside:
                pointer_x0 = x1 + 20
            else:
                pointer_x0 = x0 - 20 - new_width

            pointer_y0 = rect_center_y - (pointer.height / 2)

            # Paste the pointer on the main image
            image.paste(pointer, (int(pointer_x0), int(pointer_y0)), pointer)  # The last argument is for transparency

            # Save the combined image to a file
            image.save(f'{os.path.join(dr, str(page_num))}.png')

        # process each image-audio pair to create video chunk
        resolution = "scale=1920:-2"
        os.system(f'{args.ffmpeg} -loop 1 -i {os.path.join(dr, str(page_num))}.png -i {os.path.join(dr, audio)}.mp3 '
                  f'-vf {resolution} -c:v libx264 -tune stillimage -y -c:a aac -b:a 128k -pix_fmt yuv420p '
                  f'-shortest {os.path.join(dr, video)}.mp4')

        # ensure that there is no silence at the end of the video, and video len is the same as audio len
        os.system(f'audio_duration=$({args.ffprobe} -i {os.path.join(dr, audio)}.mp3 '
                  f'-show_entries format=duration -v quiet -of csv="p=0"); '
                  f'audio_duration=$((${{audio_duration%.*}} + 1)); '
                  f'{args.ffmpeg} -i {os.path.join(dr, video)}.mp4 -t $audio_duration '
                  f'-y -c copy {os.path.join(dr, video)}_final.mp4')

        # list of all chunks
        outvideo.write(f"file '{video}_final.mp4'\n")

    outvideo.close()

    # joint video
    os.system(f'{args.ffmpeg} -f concat -i {os.path.join(dr, "mp4_list.txt")} -y -c copy {os.path.join(dr, "output.mp4")}')

    # =============== SHORT VIDEO ====================

    if os.path.exists(os.path.join(dr, "shorts_mp3_list.txt")):

        with open(os.path.join(dr, "shorts_mp3_list.txt"), "r") as f:
            lines = f.readlines()

        # create list of chunks
        outvideo = open(os.path.join(dr, 'short_mp4_list.txt'), 'w')

        # Process each line
        for page_num, line in enumerate(lines):
            # Remove the newline character at the end of the line
            line = line.strip()

            # Split the line into components
            components = line.split()

            # The filename is the second component
            audio = components[1].replace('.mp3', '')
            video = audio.replace('-', '')

            # convert to PNG
            if page_num == 0:
                input_path = os.path.join(dr, str(page_num))
            else:
                input_path = os.path.join(dr, 'slides', f'slide_{page_num}')

            os.system(f'{args.gs} -sDEVICE=png16m -r500 -o {os.path.join(dr, str(page_num))}.png {input_path}.pdf')

            resolution = "scale=1920:-2"
            os.system(f'{args.ffmpeg} -loop 1 -i {os.path.join(dr, str(page_num))}.png -i {os.path.join(dr, audio)}.mp3 '
                      f'-vf {resolution} -c:v libx264 -tune stillimage -y -c:a aac -b:a 128k -pix_fmt yuv420p '
                      f'-shortest {os.path.join(dr, video)}.mp4')

            # ensure that there is no silence at the end of the video, and video len is the same as audio len
            os.system(f'audio_duration=$({args.ffprobe} -i {os.path.join(dr, audio)}.mp3 -show_entries format=duration '
                      f'-v quiet -of csv="p=0"); 'f'audio_duration=$((${{audio_duration%.*}} + 1)); 'f'{args.ffmpeg} '
                      f'-i {os.path.join(dr, video)}.mp4 -t $audio_duration -y -c copy {os.path.join(dr, video)}_final.mp4')

            # list of all chunks
            outvideo.write(f"file '{video}_final.mp4'\n")

        outvideo.close()

        # joint video
        os.system(f'{args.ffmpeg} -f concat -i {os.path.join(dr, "short_mp4_list.txt")} '
                  f'-y -c copy {os.path.join(dr, "output_short.mp4")}')

    # =============== QA VIDEO ====================

    if os.path.exists(os.path.join(dr, "qa_mp3_list.txt")):

        with open(os.path.join(dr, "qa_mp3_list.txt"), "r") as f:
            lines = f.readlines()

        # create list of chunks
        outvideo = open(os.path.join(dr, 'qa_mp4_list.txt'), 'w')

        qa_pages = pickle.load(open(os.path.join(dr, 'qa_pages.pkl'), 'rb'))

        # Process each line
        turn = -1
        for line_num, line in enumerate(lines):
            # Remove the newline character at the end of the line
            line = line.strip()

            # Split the line into components
            components = line.split()

            # The filename is the second component
            audio = components[1].replace('.mp3', '')
            video = audio.replace('-', '')

            # convert to PNG
            if 'question' in audio:  # question - get created slide
                turn += 1
                page_num = 0
                input_path = os.path.join(dr, 'questions', f'question_{turn}')
            else:  # answer - get single page from paper
                p_num = qa_pages[turn][page_num]
                # extract the page from PDF
                os.system(f'{args.gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={p_num+1} -dLastPage={p_num+1} -sOutputFile={os.path.join(dr, str(p_num))}.pdf {os.path.join(dr, "main.pdf")} > /dev/null 2>&1')
                input_path = os.path.join(dr, f'{p_num}')
                page_num += 1

            qa_page = 'qa_page.png'
            os.system(f'{args.gs} -sDEVICE=png16m -r500 -o {os.path.join(dr, qa_page)} {input_path}.pdf')

            resolution = "scale=1920:-2"
            os.system(f'{args.ffmpeg} -loop 1 -i {os.path.join(dr, qa_page)} -i {os.path.join(dr, audio)}.mp3 '
                      f'-vf {resolution} -c:v libx264 -tune stillimage -y -c:a aac -b:a 128k -pix_fmt yuv420p '
                      f'-shortest {os.path.join(dr, video)}.mp4')

            # ensure that there is no silence at the end of the video, and video len is the same as audio len
            os.system(f'audio_duration=$({args.ffprobe} -i {os.path.join(dr, audio)}.mp3 -show_entries format=duration '
                      f'-v quiet -of csv="p=0"); 'f'audio_duration=$((${{audio_duration%.*}} + 1)); 'f'{args.ffmpeg} '
                      f'-i {os.path.join(dr, video)}.mp4 -t $audio_duration -y -c copy {os.path.join(dr, video)}_final.mp4')

            # list of all chunks
            outvideo.write(f"file '{video}_final.mp4'\n")

        outvideo.close()

        # joint video
        os.system(f'{args.ffmpeg} -f concat -i {os.path.join(dr, "qa_mp4_list.txt")} '
                  f'-y -c copy {os.path.join(dr, "output_qa.mp4")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--paperid", type=str, default='')
    parser.add_argument("--ffmpeg", type=str, default='ffmpeg')
    parser.add_argument("--ffprobe", type=str, default='ffprobe')
    parser.add_argument("--gs", type=str, default='gs')

    args = parser.parse_args()

    main(args)