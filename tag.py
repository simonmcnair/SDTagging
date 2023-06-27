import os
import sys
import re
from typing import Mapping, Tuple, Dict
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession
import concurrent.futures
import subprocess
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm


import shutil
from collections import Counter

#pip install opencv-python pillow huggingface_hub onnxruntime
#from timer import Timer

# Needs exiftool too

#IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp')
#bmp does not support many tags
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff',)
TEXT_EXTENSIONS = ('.txt',)

class TruncatedFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.max_file_size = 20 * 1024 * 1024  # 200MB
        self.current_sequence = 0

    def emit(self, record):
        super().emit(record)
        if self.should_truncate():
            self.truncate_file()
            self.current_sequence += 1
            self.baseFilename = self.get_new_filename()

    def should_truncate(self):
        if os.path.isfile(self.baseFilename):
            return os.path.getsize(self.baseFilename) > self.max_file_size
        return False

    def truncate_file(self):
        with open(self.baseFilename, 'r+') as file:
            file.seek(self.max_file_size)
            file.truncate()

    def get_new_filename(self):
        base_name, ext = os.path.splitext(self.baseFilename)
        sequence_suffix = f'{self.current_sequence:03d}'
        return f'{base_name}_{sequence_suffix}{ext}'

def setup_logger(log_file_path, log_level=logging.INFO):
    # Create logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    

    # Create file handlers for different levels
    log_file_base = os.path.splitext(log_file_path)[0]
    
    #file_handler = TruncatedFileHandler(log_file_path)
    #debug_handler = TruncatedFileHandler(log_file_base + '_debug.log')
    #info_handler = TruncatedFileHandler(log_file_base + '_info.log')
    #warning_handler = TruncatedFileHandler(log_file_base + '_warning.log')
    #error_handler = TruncatedFileHandler(log_file_base + '_error.log')
    logfilebase = 'combo15'
    
    debug_handler = RotatingFileHandler(log_file_base + '_debug.log', mode='a', maxBytes=5*1024*1024, backupCount=4, encoding=None, delay=0)
    info_handler = RotatingFileHandler(log_file_base + '_info.log', mode='a', maxBytes=5*1024*1024, backupCount=4, encoding=None, delay=0)
    warning_handler = RotatingFileHandler(log_file_base + '_warning.log', mode='a', maxBytes=5*1024*1024, backupCount=4, encoding=None, delay=0)
    error_handler = RotatingFileHandler(log_file_base + '_error.log', mode='a', maxBytes=5*1024*1024, backupCount=4, encoding=None, delay=0)
        
    # Set log levels
    #file_handler.setLevel(logging.DEBUG)
    debug_handler.setLevel(logging.DEBUG)
    info_handler.setLevel(logging.INFO)
    warning_handler.setLevel(logging.WARNING)
    error_handler.setLevel(logging.ERROR)
    

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    #file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    

    # Add the handlers to the logger
    #logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(warning_handler)
    logger.addHandler(error_handler)

    return logger

# Example usage:
cwd = os.getcwd()
logfilepath = os.path.join(cwd, "combolog.txt")

logger = setup_logger(logfilepath)
#logger = setup_logger(logfilepath, logging.DEBUG)
#logger.debug('This is a debug message')
#logger.info('This is an info message')
#logger.warning('This is a warning message')
#logger.error('This is an error message')


# noinspection PyUnresolvedReferences
def image_make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# noinspection PyUnresolvedReferences
def image_smart_resize(img, size):
    # Assumes the image has already gone through image_make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    else:  # just do nothing
        pass

    return img

class WaifuDiffusionInterrogator:
    def __init__(
            self,
            repo='SmilingWolf/wd-v1-4-vit-tagger-v2',
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            mode: str = "auto"
    ) -> None:
        self.__repo = repo
        self.__model_path = model_path
        self.__tags_path = tags_path
        self._provider_mode = mode

        self.__initialized = False
        self._model, self._tags = None, None

    def _init(self) -> None:
        if self.__initialized:
            return

        model_path = hf_hub_download(self.__repo, filename=self.__model_path)
        tags_path = hf_hub_download(self.__repo, filename=self.__tags_path)

        self._model = InferenceSession(str(model_path))
        self._tags = pd.read_csv(tags_path)

        self.__initialized = True

    def _calculation(self, image: Image.Image)  -> pd.DataFrame:
        self._init()

        _, height, _, _ = self._model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = image_make_square(image, height)
        image = image_smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        confidence = self._model.run([label_name], {input_name: image})[0]

        full_tags = self._tags[['name', 'category']].copy()
        full_tags['confidence'] = confidence[0]

        return full_tags

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        full_tags = self._calculation(image)

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(full_tags[full_tags['category'] == 9][['name', 'confidence']].values)

        # rest are regular tags
        tags = dict(full_tags[full_tags['category'] != 9][['name', 'confidence']].values)

        return ratings, tags

WAIFU_MODELS: Mapping[str, WaifuDiffusionInterrogator] = {
    'wd14-vit-v2': WaifuDiffusionInterrogator(),
    'wd14-convnext': WaifuDiffusionInterrogator(
        repo='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
}
RE_SPECIAL = re.compile(r'([\\()])')


def image_to_wd14_tags(filename, image:Image.Image) \
        -> Tuple[Mapping[str, float], str, Mapping[str, float]]:
    try:
        model = WAIFU_MODELS['wd14-vit-v2']
        ratings, tags = model.interrogate(image)

        filtered_tags = {
            tag: score for tag, score in tags.items()
            if score >= .35
        }

        text_items = []
        tags_pairs = filtered_tags.items()
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
        for tag, score in tags_pairs:
            tag_outformat = tag
            tag_outformat = tag_outformat.replace('_', ' ')
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
            text_items.append(tag_outformat)
        output_text = ', '.join(text_items)

        return ratings, output_text, filtered_tags
    except Exception as e:
        logger.error("Exception getting tags from image " + filename + ". " + str(e))

def check_and_del_text_file(file_path, words):
    # Check if the file exists
    try:
        if not os.path.isfile(file_path):
            # If the file doesn't exist, create it and write the words
            logger.info("check_and_del_text_file: " + "No text metadata file exists.  Great.  Awesome.  Super.  Smashing.")
            return True
        else:
            # Read the contents of the text file
            logger.info("check_and_del_text_file: " + file_path + " exists.  Checking for text")
            with open(file_path, 'r') as file:
                file_contents = file.read()

            if "," in file_contents:
            # Split the file contents into individual words
                file_words = set(file_contents.strip().split(','))
                
                # Split the input words into individual words
                input_words = set(words.strip().split(','))
                logger.info(file_path + "Words detected from ML are " + words)
                logger.info(file_path + "Words detected from TXT are " + file_contents)
                
                # Check if all the input words are present in the file words
                if not input_words.issubset(file_words):
                    # Append the input words to the file
                    logger.error("check_and_del_text_file: " + file_path + " contains words: " + str(file_contents) + " but image file contains " + str(words))
                    return False    
                else:
                    logger.info("check_and_del_text_file: " + "Words required and present are : " + words)
                    logger.info("check_and_del_text_file: " + "All words already present in " + file_path + " Delete the file")
                    delete_file(file_path)
                    return True
            else:
                logger.info("check_and_del_text_file: " + file_path + " is not a csv file.  Skipping")
                return True
    except Exception as e:
        logger.error("Exception check_and_del_text_file: " + "Error for " + file_path + ". Retcode: " + str(e.returncode) + " check and del text file:" + str(e.output) + ".")
        return False
    
def check_and_append_text_file(file_path, words):
    # Check if the file exists
    try:
        if not os.path.isfile(file_path):
            # If the file doesn't exist, create it and write the words
            logger.info("check_and_append_text_file: Creating file " + file_path)
            with open(file_path, 'w') as file:
                file.write(words)
        else:
            # Read the contents of the text file
            with open(file_path, 'r') as file:
                file_contents = file.read()

            if ',' in file_contents:
                # Split the file contents into individual words
                file_words = set(file_contents.strip().split(','))

                # Split the input words into individual words
                input_words = set(words.strip().split(','))

                # Check if all the input words are present in the file words
                if not input_words.issubset(file_words):
                    # Append the input words to the file
                    logger.info("check_and_append_text_file: appending " + str(words) + " to " + file_path)
                    with open(file_path, 'a') as file:
                        file.write(',' + words)
                else:
                    logger.info("check_and_append_text_file: All words already present in " + file_path)
            else:
                logger.info("check_and_append_text_file: " + file_path + " is not a CSV file")
                return True
            return True

    except Exception as e:
        logger.error("check_and_append_text_file: " + file_path + ".  " + str(words) + ".  error " + str(e))
        return False
def delete_file(file_path):
    if '.txt' in file_path.lower():
        try:
            os.remove(file_path)
            logger.info("Deleted file: " + file_path)
            return True
        except OSError as e:
            logger.error("Exception deleting file: " + file_path + ". " + str(e))
            return False
        except Exception as e:
            logger.error("Exception: " + str(e.returncode) + ".  " + str(e.output) + ".  From " + file_path)
            return False
        
    else:
        logger.info(file_path + ".  Can only delete text files")
        return True

def move_file_to_prefixed_folder(filepath, text_string):
    cwd = os.getcwd()
    abs_filepath = os.path.abspath(filepath)
    rel_filepath = os.path.relpath(abs_filepath, cwd)

    filename = os.path.basename(rel_filepath)
    new_folder_path = os.path.join(text_string, os.path.dirname(rel_filepath))
    new_file_path = os.path.join(new_folder_path, filename)

    logger.info("move_file_to_prefixed_folder: Moved " + filepath + " to " + new_file_path)

    os.makedirs(new_folder_path, exist_ok=True)
    shutil.move(rel_filepath, new_file_path)

def exiftool_del_dupetags(path):
    logger.info("exiftool_del_dupetags: " + path + ": Removing duplicate tags")
    try:
        output = subprocess.check_output(['exiftool', \
                                              '-overwrite_original' , \
                                                '-P', \
                                                '-XMP:Subject<${XMP:Subject;NoDups(1)}', \
                                                '-IPTC:Keywords<${IPTC:Keywords;NoDups(1)}', \
                                                '-XMP:CatalogSets<${XMP:CatalogSets;NoDups(1)}', \
                                                '-XMP:TagsList<${XMP:TagsList;NoDups(1)}', \
                                                path], \
                                                stderr=subprocess.STDOUT, universal_newlines=True)
        logger.info("exiftool_del_dupetags MODIFY success : " + path + ". output: " + output)
    except Exception as e:
        logger.error("Exception in exiftool_del_dupetags : " + path + ". Error: " + str(e))
        return False

    return True

def exiftool_copy_XMPSubject_to_TagsList(path):
    logger.info("exiftool_copy_tags_to_TagsList: " + path + ": Removing duplicate tags")
    try:
        output_xmp = subprocess.check_output(['exiftool', '-overwrite_original' ,'-P', '-sep "##"', '-XMP:TagsList<${XMP:Subject;NoDups(1)}', path], stderr=subprocess.STDOUT, universal_newlines=True)
        logger.info("exiftool_copy_tags_to_TagsList MODIFY success XMP: " + path + ". output: " + output_xmp)
    except Exception as e:
        logger.error("Exception in exiftool_copy_tags_to_TagsList XMP: " + path + ". Error: " + str(e))

    
def exiftool_is_photo_tagged(photo_path):
    try:
        output = subprocess.check_output(['exiftool', '-P', '-s', '-XMP-acdsee:tagged', photo_path]).decode().strip()
        #logger.info("output: " + output)
        if 'true' in output.lower():
            #logger.info(photo_path + " already tagged")
            return True
        else:
            logger.info(photo_path + "  is not tagged as processed. " + str(output))
            return False
    except Exception as e:
        logger.error("Exception exiftool_is_photo_tagged: "+ photo_path + ".  Error " + str(e.returncode) + ".  " + str(e.output) + ".")
        #move_file_to_prefixed_folder(photo_path, 'badfiles')
        return False

def exiftool_make_photo_tagged(is_tagged, photo_path):
    try:
        if not exiftool_is_photo_tagged(photo_path) :
            output = subprocess.check_output(['exiftool', '-overwrite_original', '-P', '-s', '-XMP-acdsee:tagged=' + str(is_tagged), photo_path]).decode().strip()
            logger.info(photo_path + ".  Wasn't tagged. trying to tag as " + str(is_tagged) + " !  Output: " + output)
            if 'updated' in output.lower():
                logger.info(photo_path + ".  successfully  MODIFY tagged as " + str(is_tagged) + " !  Output: " + output)
                return True
            else:
                logger.error("Failed to change photo " + photo_path + " tagged to  " + str(is_tagged))
                return False
        else:
            logger.info(photo_path + " is already tagged.  Not modifying")
            return True
    except Exception as e:
        logger.error("Exception " + str(e.returncode) + ".  " + str(e.output) + ".  From " + photo_path)
        return False
def exiftool_batch_untag(path):
    try:
            output = subprocess.check_output(['exiftool', '-overwrite_original', '-P', '-s', '-XMP-acdsee:tagged=False', '-r' ,path])
            logger.info(path + ".  Wasn't tagged.  trying to tag as False !  Output: " + output)
            if 'updated' in output.lower():
                logger.info(path + ".  successfully  MODIFY tagged as False !  Output: " + output)
                return True
            else:
                return False

    except Exception as e:
        logger.error("Exception " + str(e.returncode) + ".  " + str(e.output) + ".  From " + path)
        return False


def exiftool_get_existing_tags(img_path):
    try:
        tags_dict = {
            'XMP:Subject': [],
            'IPTC:Keywords': [],
            'XMP:CatalogSets': [],
            'XMP:TagsList': []
        }

        loopcounter = 1
        Process = True
        check1 = False
        check2 = False
        removespaces = False
        while Process:
            existing_tags = subprocess.check_output(['exiftool', '-XMP:Subject', '-IPTC:Keywords', '-XMP:CatalogSets', '-XMP:TagsList', img_path]).decode().strip()
            logger.debug(img_path + ": exiftool_get_existing_tags exif output: \n" + existing_tags)
            separator = '\n'
            print("LOOP " + str(loopcounter) + " for " + img_path)
            loopcounter+=1
           
            if not existing_tags:
                print("image has no tags")
                return None
            else:
                if '\r\n' in existing_tags:
                    print("exiftool_get_existing_tags: rn detected. Windows?")
                    separator = '\r\n'

                if removespaces == True:
                    ret = subprocess.check_output(['exiftool','-P','-overwrite_original', '-api', '"Filter=s/^ +//"','-TagsFromFile','@','-subject','-XMP:subject','-IPTC:Keywords','-XMP:CatalogSets','-XMP:TagsList',img_path])
                    logger.info("output was " + str(ret) + " for " + img_path)
                    
                for tag in existing_tags.split(separator):
                    logger.debug(img_path + ": exiftool_get_existing_tags : tag=" + tag)
                    for tag_type in tags_dict.keys():
                        new = tag_type.split(':')[1]
                        if new == 'CatalogSets':
                            new ='Catalog Sets'
                        if new == 'TagsList':
                            new ='Tags List'
                        logger.debug(img_path + ": exiftool_get_existing_tags looking for " + new)
                        if tag.startswith(new):
                            tag_value = tag.split(':', 1)[1].strip()  # Split using the first colon only
                            
                        #    occurrences = tag_value.count(",  ")
                        #    if occurrences >0:
                        #        logger.info("!!!!!!!!!!!!!!!!!!!!!!!!! multiple spaces !!!!!!!!!!!!!!!!!  Number of occurrences:" + str(occurrences) + ".  Image is " + img_path)
                        #        print(      "!!!!!!!!!!!!!!!!!!!!!!!!! multiple spaces !!!!!!!!!!!!!!!!!  Number of occurrences:" + str(occurrences) + ".  Image is " + img_path)
                        #        removespaces = True
                        #   else:
                            check1 = True
                            
                        # occurrences = tag_value.count(", ")
                        # if occurrences >0:
                        #     logger.info("!!!!!!!!!!!!!!!!!!!!!!!!! one leading space !!!!!!!!!!!!!!!!!  Number of occurrences:" + str(occurrences) + ".  Image is " + img_path)
                        #     print(      "!!!!!!!!!!!!!!!!!!!!!!!!! one leading space !!!!!!!!!!!!!!!!!  Number of occurrences:" + str(occurrences) + ".  Image is " + img_path)
                        #     removespaces = True
                        # else:
                            check2 = True
                                
                            if check1 == True and check2 == True:
                                logger.info("No space padding for file " + img_path + ".  Continuing")
                                Process = False
                            tags_dict[tag_type].extend(tag_value.split(', '))
                            #tags_dict[tag_type].extend([tag.strip() for tag in tag_value.split(',')])

        logger.debug(img_path + ". exiftool_get_existing_tags Exiftool output XMP:Subject      :" + str(tags_dict['XMP:Subject']))
        logger.debug(img_path + ". exiftool_get_existing_tags Exiftool output IPTC:Keywords    :" + str(tags_dict['IPTC:Keywords']))
        logger.debug(img_path + ". exiftool_get_existing_tags Exiftool output XMP:CatalogSets :" + str(tags_dict['XMP:CatalogSets']))
        logger.debug(img_path + ". exiftool_get_existing_tags Exiftool output XMP:TagsList    :" + str(tags_dict['XMP:TagsList']))

        return tags_dict

    except Exception as e:
        logger.error("Exception in exiftool_get_existing_tags: " + img_path + ". Error: " + str(e))
        return {}
    
def exiftool_Update_tags(img_path, tags):
    cmd = ['exiftool', '-overwrite_original', '-P']
    try:
       existing_tags = exiftool_get_existing_tags(img_path)
    except Exception as e:
        logger.error("Exception in exiftool_Update_tags: " + img_path + ".  error " + str(e))
        return False

    try:
        updated = False
        for tag_type, existing_tags_list in existing_tags.items():
            for tag in tags:
                tag = tag.strip()
                if tag and tag not in existing_tags_list:
                    logger.debug("exiftool_Update_tags: need to add " + tag_type + " field " + tag + " to " + img_path)
                    #cmd.append(f'-{tag_type}:{tag_type}+={tag}')
                    cmd.append(f'-{tag_type}-="{tag}" -{tag_type}+="{tag}"')
                    updated = True

        if updated:
 #           logger.debug("Wierd thing I don't understand : " + ['-' + tag_type + ':' + tag_type + '+=' + tag for tag_type in existing_tags.keys() for tag in tags])
            cmd.extend(['-' + tag_type + '+=' + tag for tag_type in existing_tags.keys() for tag in tags])
            cmd.append(img_path)
            try:
                ret = subprocess.run(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
                logger.debug("exiftool_Update_tags command line was " +str(cmd))
                logger.info("exiftool_Update_tags  MODIFY" + img_path + ".  Exiftool update completed successfully.")
                logger.debug("exiftool_Update_tags  MODIFY" + img_path + ".  Exiftool update completed successfully.  " + str(ret))
                return True
            except Exception as e:
                logger.error("Exception in exiftool_Update_tags: " + img_path + ".  error " + str(e))
                return False
        else:
            logger.info(img_path + ":  exiftool_Update_tags.  Nothing to do, tags (" + str(tags) + ")are correct")
            return True 

    except Exception as e:
        logger.error("Exception in exiftool_Update_tags: " + img_path + ".  error " + str(e))
        return False


def are_tags_correct(img_path, tags):
    try:
        existing_tags = exiftool_get_existing_tags(img_path)

        tags_dict = {
            'XMP:Subject': [],
            'IPTC:Keywords': [],
            'XMP:CatalogSets': [],
            'XMP:TagsList': []
        }

        for stag in tags:
            for tag_type in tags_dict.keys():
                tags_dict[tag_type].append(stag.strip())
        #logger.debug("TEST " +  str(tags_dict))

        aretagscorrect = True
        logger.debug("validate tags for " + img_path + ".  tags to check against are " + str(tags))
        logger.debug("validate tags for " + img_path + ".  tags from exiftool_get_existing_tags are " + str(existing_tags))
        for tag_type, existing_tags_list in existing_tags.items():
            
            for tagentry in tags_dict[tag_type]:
                if tagentry not in existing_tags_list:
                    logger.error("are_tags_correct: " + img_path + "." + tag_type + "." + tagentry + " is missing.")
                    aretagscorrect = False
                #else:
                #    logger.debug("are_tags_correct: " + img_path + "." + tag_type + "." + tagentry + " is present.")
        if aretagscorrect==False:
            return False
        else:
            return True

    except Exception as e:
        logger.error("Exception in are_tags_correct: " + img_path + ". Error " + str(e.returncode) + " removing duplicate tags: " + str(e.output) + ".")
        return False
    
def find_duplicate_tags_in_file(img_path):
    try:
        existing_tags = exiftool_get_existing_tags(img_path)
        duplicate_tags = {}

        for tag_type, existing_tags_list in existing_tags.items():
            #logger.debug(img_path + " find_duplicate_tags_in_file tag_type:" + tag_type + " .  existing_tags_list:" + str(existing_tags_list))
            for tag in existing_tags_list:
                #logger.debug(img_path + " find_duplicate_tags_in_file tag:" + tag_type + ":" + tag)
                if tag and existing_tags_list.count(tag) > 1:
                    if tag not in duplicate_tags:
                        duplicate_tags[tag] = {
                            'count': existing_tags_list.count(tag),
                            'tag_type': [tag_type]
                        }
                    else:
                        duplicate_tags[tag]['count'] += existing_tags_list.count(tag)
                        duplicate_tags[tag]['tag_type'].append(tag_type)

        if duplicate_tags:
            logger.info("find_duplicate_tags_in_file: Duplicate tags found in " + img_path)
            for tag, info in duplicate_tags.items():
                logger.debug("find_duplicate_tags_in_file: Duplicate Tags in %s: %s, Count: %s, Tag Types: %s", img_path, tag, info['count'], ', '.join(set(info['tag_type'])))
            return True

        else:
            logger.info("find_duplicate_tags_in_file: No duplicate tags found in " + img_path)
            return False

    except Exception as e:
        logger.error("find_duplicate_tags_in_file Exception:" + str(e))


def process_file(image_path):
    #image_path = 'C:\\Users\\Simon\\Downloads\\w6bgPUV.png'
    reprocess = False
    logger.info("Processfile " + " START Processing " + image_path)
    output_file = os.path.splitext(image_path)[0] + ".txt"


    if exiftool_is_photo_tagged(image_path) and not reprocess:
        logger.info(image_path + " is already tagged")
        if find_duplicate_tags_in_file(image_path) :
            logger.debug("Processfile " +  "There were duplicate tags in " + image_path)
            exiftool_del_dupetags(image_path)
        else:
           logger.debug("Processfile " +  "There were no duplicate tags in " + image_path)
           
        if  os.path.isfile(output_file):
            logger.info(image_path + ".  Need to process as there is a " + output_file + " file which could be deleted.")
        else:
            logger.info(image_path + ".  File is tagged as processed.  No dupe tags.  No txt file.  Finished.  Success.")
            return True

    else:
        logger.info("Processfile " + image_path + " not marked as processed.  Continue processing ")

    try:
        image = Image.open(image_path)
        logger.info("image: " + image_path + " successfully opened.  Continue processing ")
    except Exception as e:
        logger.error("Processfile Exception1: " + " failed to open image : " + image_path + ". FAILED Error: " + str(e) + ".  Skipping")
        move_file_to_prefixed_folder(image_path, 'badfiles')
        return False

    try:
        gr_ratings, gr_output_text, gr_tags = image_to_wd14_tags(image_path, image)
        #gr_output_text = gr_output_text + ',tagged'
        tagdict = gr_output_text.split(",")
        logger.info("Processfile tag extract success. " + image_path + ".  caption: " + gr_output_text)
    except Exception as e:
        logger.error("Processfile tag extraction for " + image_path + " didn't work. FAILED  Skipping")
        return False

    try:
        tagdict = [substr for substr in tagdict if substr]
    except Exception as e:
        logger.error("Processfile tagdict substr Error. FAILED  Well that didn't work.")
        return False

    try:
        ret =  exiftool_Update_tags(image_path, tagdict)
        if ret == True:
            logger.info("exiftool_Update_tags success. " + image_path + ".")
        else:
            logger.error("exiftool_Update_tags FAILED. " + image_path + ".")
            return False
    except Exception as e:
        logger.error("Processfile exiftool_Update_tags FAILED Exception. " + image_path + ". " + str(e) )
        return False
    
    try:
        ret = are_tags_correct(image_path, tagdict)
        if ret == True:
            logger.info(image_path + " tags added correctly " + str(ret))
        else:
            logger.error(image_path + " tags NOT added correctly. FAILED " + str(ret))
            return False
    except Exception as e:
        logger.error("Processfile are_tags_correct FAILED Exception " + ". " + image_path + ". " + str(e) )
        return False
    
    try:
        ret = check_and_del_text_file(output_file,gr_output_text)
        if ret == True:
            logger.info(image_path + " check_and_del_text_file success " + str(ret))
        else:
            logger.error(image_path + " check_and_del_text_file FAILED.  Not marking tagged " + str(ret))
            return False
    except Exception as e:
        logger.error("Processfile check_and_del_text_file FAILED.  Not marking as tagged. Exception. " + ". " + image_path + ". " + str(e) )
        return False

    try:
        logger.info(image_path + ".  If I got here then previous steps were successful.  Mark as processed")      
        ret = exiftool_make_photo_tagged('True',image_path)
        if ret == True:
            logger.info("Processfile " + " SUCCESS marking as processed " + image_path + ". " + str(ret))
            return True
        else:
            logger.error("Processfile " + " FAILED marking as processed " + image_path + ". " + str(ret))
            return False       
    except Exception as e:
        logger.error("Processfile exiftool_make_photo_tagged FAILED Exception. " + ". " + image_path + ". " + str(e) )
        return False

def Add_a_Tag(image_path, tag):
    print("Attempting to add tags " + str(tag))
    retval = exiftool_Update_tags(image_path, tag)
    if find_duplicate_tags_in_file(image_path) :
        logger.debug("Processfile " +  "There were duplicate tags in " + image_path)
        exiftool_del_dupetags(image_path)


def process_images_in_directory(directory, tag,person):
    # Process each image in the directory
    image_paths = []
    overall_processed_images = 0
    logger.info("Starting")

    logger.info("fetching file list.  This could take a while.")
    for root, dirs, files in os.walk(directory):
#    for root, dirs, files in sorted(os.walk(directory)):
        #files.sort()
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(IMAGE_EXTENSIONS):
                # Get the full path to the image
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                print("Adding " + image_path)
    logger.info("Done.Array created")

    image_paths.sort()  # Sort the filepaths based on base filenames
    #image_paths.sort(key=lambda x: os.path.basename(x))  # Sort the filepaths based on base filenames

    num_images = len(image_paths)
    logger.info("number of images to process is " + str(num_images))
    processed_images = 0
    average_time_per_image = 0

#    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        completed_count = 0
        cnt = 1
        # Submit the image processing tasks
        for image_path in image_paths:
            print(str(cnt) + "." + image_path)
            cnt +=1
            if tag == "" and person == False:
                print("Processing as normal")
#                future = executor.submit(process_file, image_path)
#                futures.append(future)
            if person == True:
                print("Add Folder as a person")
                #parent_directory = os.path.dirname(image_path)
                parent_directory = os.path.basename(os.path.dirname(image_path))
                person2 =[]
                person2.append("Person/" + parent_directory)
                print("Add " + str(person2) + " to file " + image_path + " and process as adding parent folder as person tag")
                future = executor.submit(Add_a_Tag, image_path, person2)
                futures.append(future)                
            if tag != "":
                print("Adding a tag to a file")
#                future = executor.submit(Add_a_Tag, image_path, tag)
#                futures.append(future)


    # Use tqdm to display progress bar
        with tqdm(total=len(futures)) as pbar:
            while completed_count < len(futures):
                completed_count = sum(1 for future in futures if future.done())
                pbar.update(completed_count - pbar.n)


    logger.info("finished")

    #            processed_images += 1
    #            overall_processed_images += 1
    #            overallfolderprogress = overall_processed_images / num_images

    #            if future.done():
    #                completed_count += 1
    #                result = future.result()
    #                print(f"Image processed: {result} ({completed_count}/{total_count})")
            # Check if any futures are completed or running
    #    eta = (num_images - processed_images) * average_time_per_image
        
        #for future in concurrent.futures.:

# Specify the directory containing the images

# Process the images in the directory and generate captions
#process_images_in_directory(image_directory)

def execute_script(directory=None, tag=None, person=None):
    if directory is None:
        if os.name == 'nt':  # Windows
            #directory = r'X:\\Stable\\dif\\stable-diffusion-webui-docker\\output'
            directory = r'Z:\Stable\dif\stable-diffusion-webui-docker\output\\'
        else:  # Linux or macOS
            directory = '/srv/dev-disk-by-uuid-e83913b3-e590-4dc8-9b63-ce0bdbe56ee9/Stable/dif/stable-diffusion-webui-docker/output'

    if tag is None:
        taglist = ""  # Default value if no tag is provided
    else:
        taglist = tag.split(",")

    if person is None:
    #remember to change back
        print("executing default of disabled")
        personopt = False  # Default value if no tag is provided
    elif person == 'True':
        print("Person and personopt set to true")
        personopt = True
    elif person == 'False':
        print("person and personopt set to false")
        personopt = False
    else:
        print("error")
        exit

    

    # Change the current working directory to the specified directory
    #os.chdir(directory)

    # Execute your script here
    # For demonstration purposes, let's print the current working directory
    #logger.info("Current working directory:", os.getcwd())
    process_images_in_directory(directory, taglist,personopt)
    logger.info("Processing complete!")

def execute_single(file, tag=None):
    if tag is None:
        taglist = ""  # Default value if no tag is provided
        process_file(file)
    else:
        print("tag provided:" + str(tag))
        taglist = tag.split(",")
        Add_a_Tag(file,taglist)
    logger.info("Processing complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Opts on command line " + str(len(sys.argv)))
        # Use the path provided as a command line argument
        print("path command line " + str(sys.argv[1]))
        print("tag arg on command line " + str(sys.argv[2]))
        print("tag person command line " + str(sys.argv[3]))

        path_arg = sys.argv[1]
        tag_arg = sys.argv[2] if len(sys.argv) >= 2 else None
        tag_person = sys.argv[3] if len(sys.argv) >= 3 else None
        abs_path = os.path.abspath(path_arg)

        print("path: " + abs_path)
        if os.path.isdir(abs_path):
            # The provided argument is a directory
            execute_script(directory=abs_path, tag=tag_arg,person=tag_person)
        elif os.path.isfile(abs_path):
            # The provided argument is a file
            execute_single(abs_path, tag=tag_arg)
        else:
            print("Invalid path argument. Please provide a valid directory or file.")
            sys.exit(1)
    else:
        # Use the predefined directory if no command line argument is provided
        print("no opts on command line")
        execute_script()
