import numpy as np
import re

def get_ccd_id_number(image):
    """ returns the ccd name/id from the input fits file
    """

    if image.__contains__("_ccd"):
        try:
            id_ccd = int(image.split('_ccd')[-1].split('_')[0])
            return id_ccd
        except ValueError:
            return None

    image_file_name = image.split("/")[-1]
    if image_file_name.startswith("avg_"):
        try:
            id_ccd = int(image.split("/")[-1].split(".")[-2].split("_")[-4])
            return id_ccd
        except ValueError:
            return None

    return None

def get_image_and_run_number(fname):
    """Function to get the RUN and IMAGE number from the file name

        XXX Hardcoded for Compton file names XXX

    """
    image = fname.split("/")[-1]
    if image.startswith("avg_skips"):
        # assuming data for the new ACM background run
        print("Image IDs following 'LBC Background Run with ACM (23/08/2024)' pattern")
        isSource = False
        id_run = 0
        try:
            id_image = int(image.split(".")[0].split("_")[-1])
        except ValueError:
            id_image = 0
        return id_image,id_run,None

    elif image.__contains__("Source"):
        print("Image and Run IDs following 'Source' Compton pattern\n")
        img_type = "_Source_"
        isSource = True
    elif image.__contains__("Bkg_"):
        print("Image and Run IDs following 'Bkg_' Compton pattern\n")
        img_type = "_Bkg_"
        isSource = False
    elif image.__contains__("Image_comm_"):
        print("Image and Run IDs following 'Image_comm' Compton pattern\n")
        isSource = True
        try:
            id_run = int(image.split('/')[-1].split('_')[-2])
        except ValueError:
            id_run = np.random.randint(20211217)

        try:
            id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        except ValueError:
            id_image = np.random.randint(20211217)

        return id_image,id_run,isSource

    elif image.__contains__("clean_"):
        print("Image and Run IDs following 'clean' Compton pattern\n")
        isSource = False
        try:
            id_image = int(image.split(".")[0].split("_")[-1])
        except ValueError:
            id_image = np.random.randint(20211217)

        return id_image,0,isSource

    elif image.__contains__("skip_") or image.__contains__("bkg_"):
        print("Image and Run IDs following 'skip_ && bkg_' LBC-Leach pattern\n")
        isSource = False
        image = re.sub("_ext[0-9]_waders.fits",".fits",image)
        image = re.sub("_waders_waders.fits",".fits",image)
        image = re.sub("_waders.fits",".fits",image)
        image = re.sub("_pedestal_subtracted","",image)
        image = re.sub("_compressed","",image)

        try:
            id_image = int(image.split(".")[0].split("_")[-1])
        except ValueError:
            id_image = np.random.randint(20211217)
        try:
            id_run = int(int(image.split(".")[0].split("_")[-1])//100)
        except ValueError:
            id_run = np.random.randint(20211217)

        return id_image,id_run,isSource

    elif image.__contains__("proc_") and image.__contains__("_modS89") and image.__contains__("_img"):
        print("Image and Run IDs following 'proc_ && _modS89_ && _img' MOSKITA pattern\n")
        isSource = False
        try:
            id_image = int(image.split("_img")[1].split(".")[0])
        except ValueError:
            id_image = 0
        return id_image,0,isSource

    elif image.__contains__("skp_") or image.__contains__("_LTA_") or image.__contains__("_lta_"):
        print("Image and Run IDs following 'skp_ && _lta_' LBC-LTA pattern\n")
        isSource = False
        try:
            id_image = int(image.split(".")[0].split("_")[-1])
        except ValueError:
            id_image = np.random.randint(20211217)

        try:
            id_run = int(image.split(".")[0].split("_")[-3])
        except ValueError:
            id_run = np.random.randint(20211217)

        return id_image,id_run,isSource

    elif image.startswith("avg_"):
        try:
            id_image = int(image.split("/")[-1].split(".")[-2].split("_")[-1])
        except ValueError:
            id_image = 0
        return id_image,0,False

    else:
        print("Do not match any patter fiile\n")
        return 0,0,False


    id_run   = image.split("/")[-1].split(img_type)[-1].split("_")[0]

    # compressed images: DATA processed by panaSKImg willc ontain compressed on his file name
    if image.__contains__("compressed"):
        id_image = image.split("/")[-1].split("_Source_")[-1].split("_")[-2]
    else:
        id_image = image.split("/")[-1].split(img_type)[-1].split("_")[-1].split(".")[0]

    ### added to extract the value from Compton data file format
    try:
        id_image_int = int(id_image)
    except ValueError:
        id_image = image.split("/")[-1].split(img_type)[-1].split("_")[1]
        # if is still not an integer, just set to 0
        try:
            id_image = int(id_image)
        except ValueError:
            id_image = 0

    ### added for those image number that are larger than the double
    if int(id_image) > 100000:
        id_image = id_image[:5]

    ### check that id_run and id_image is a number
    try:
        id_image = int(id_image)
    except (TypeError,ValueError):
        id_image = 0

    try:
        id_run = int(id_run)
    except (TypeError,ValueError):
        id_run = 0

    return id_image,id_run,isSource
