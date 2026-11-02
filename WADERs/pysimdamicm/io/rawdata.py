""":mod:`pydqm4lbc.rawdata`
    
    XXX Description of the moduele XXX
    
.. moduleauthor:: Nuria Castello-Mor
"""
from pysimdamicm.utils.config import get_default_configuration
config_fits_raw_image = get_default_configuration(False)
from pysimdamicm.setups.data_skipper_corrections import CorrectLeachBugProcess, InvertPolarity
from pysimdamicm.utils.units import Units
u=Units()

from abc import ABCMeta, abstractmethod
import numpy as np
from astropy.io import fits
import warnings
warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)

from matplotlib  import pyplot as plt
from os.path import abspath
import dateutil

__verbose__ = False

class RawData(metaclass=ABCMeta):
    """
    """
    def __init__(self,file_name):
        self.isdata = True
        self._NDCM = 1
       
        # several parameters will be generated along the DQM process, all headers from files in this list will be
        # at the end updated
        self.fnames_to_update_header = []

        ### is absolute path??
        if file_name[0] != "/":
            file_name = abspath(file_name)
        self.file_name = file_name
        self._file_name = file_name.split("/")[-1]
        self.__type__ = self._file_name.split(".")[-1]
        self._directory = "/"+"/".join(file_name.split("/")[1:-1])
        self.output = None

        ### DATA FROM SERIAL REGISTER (NEGATIVE NUMBERING) OR FROM THE ACTIVE REGION
        # if the data contains only the serial register data (name has 'serial'), the CCD number
        # will be 0
        # SERIAL, and CCD data will be defined in the ME amplifier using the direciton of the movement of the charge
        self.ccd = 1

        self.__process_chain__ = ''
        self.killed = False
        self.compressed = False
        
        self.figure_id = 0
        
        # define an empty list to append PNG plot files related to him to be linked in the mongoDB
        self.wr_darkcurrent = []
        self.wr_overscan    = []
        
        # some conventional parameters
        self.ampdir  = 1
        self.itgtime = 1
        self.npbin = 1
        self.nsbin = 1
        self.exposure_time = 1
        self.read_time = 1

        self.data_member_type = {
                "nskips": int, 
                "ncols":int, 
                "nrows":int, 
                "npbin":int, 
                "nsbin":int,
                "amp":str,
                "exposure_time":float,
                "read_time":float}

    def info(self):
        import types
        for attr in dir(self):
            val = getattr(self,attr)
            if attr[0] != '_' and not isinstance(val, types.MethodType):
                if attr not in ['data_member_type','image_header','image' ,'killed']:
                    print(" * {} = {} ".format(attr,val))
        if hasattr(self,'image'):
            print( " * image.shape = ", self.image.shape)
        if hasattr(self,'image_mean_compressed'):
            print( " * image_mean_compressed.shpae ", self.image_mean_compressed.shape )

    @property
    def output(self):
        return self.__output
    @output.setter
    def output(self,val):
        if val is not None:
            ##### assuming is coming from DQM
            # directory for 
            #   avgimg (only for compressed images), logs (not relevant here), me (only for me
            #       plots), and others (debug and others plots/fits)
            self.__output  = "{}/{}_".format(val['others'],self._file_name.split(".")[0])
            try:
                self.me_output = val['me']+"/"
            except KeyError:
                self.me_output = self._directory+"/"
            try:
                self.avg_output = "{}/{}_".format(val['avg'],self._file_name.split(".")[0])
            except KeyError:
                self.avg_output = self.__output
            try:
                self.run = val['run']
            except KeyError:
                self.run = self.__output

            for subdir in ['me_pcd','me_dcfit','me_sed','me_ect','me_img','me_recon']:
                try:
                    setattr(self, subdir, val[subdir]+"/")
                except KeyError:
                    setattr(self, subdir, self._directory+"/")

        else:
            self.__output = "{}/{}_".format(self._directory,self._file_name.split(".")[0])
            self.me_output = self._directory+"/"
            self.avg_output = self.__output
    
    @abstractmethod
    def get_datetime(self):
        raise NotImplementedError()

    @abstractmethod
    def get_from_header(self):
        raise NotImplementedError()

    #@abstractmethod
    #def get_img_sensor(self):
    #    raise NotImplementedError()
    
    #@abstractmethod
    #def get_img_overscan(self):
    #    raise NotImplementedError()

    @abstractmethod
    def get_overscan_and_sensor_dimensions():
        raise NotImplementedError()

    def prepare_data(self):
        """
        """
        print("RawData INFO.")
        print(" *********************************************************************** ")
        print(" * Define mask for sensor and over/pre-scan regions '{}'".format(self._file_name.split("/")[-1]))
        print(" *********************************************************************** ")
        self.get_overscan_and_sensor_dimensions()
        self.print_info_ccd_region()
        print(" ***********************************************************************. \n")
        return
    
    def get_image_attribute_name(self,image_attr):
        """
        """
        if image_attr != 'raw':
            image_attribute_name = 'image_{}'.format(image_attr)
        else:
            image_attribute_name = 'image'
        
        if not hasattr(self,image_attribute_name):
            raise IOError(f"Attribute {image_attr} is not found in FitsRawData Object: check 'image' params on your json file")

        return  image_attribute_name
    
    def get_skip_image_and_broadcasted_mask(self,image_attr='raw'):
        """Build a 3D masked image, where axis 2 is the skip axis masked with the active region 
        """
        itype = self.get_image_attribute_name('raw')
        image = getattr(self,itype).copy()
        for attr in ['mask_image_overscan_cols', 'mask_image_prescan_cols', 'mask_image_active_region','rows','cols','mask_full_extension']:
            setattr(self,attr,self.amplifier[self.execute_process_in_amp][attr])
         
        mask_amp  = self.amplifier[self.execute_process_in_amp]['mask_image_active_region'].copy()
        mask_amp  = np.broadcast_to( np.expand_dims(mask_amp, axis=-1), image.shape )

        image_amp = np.ma.array(image, mask=mask_amp)

        return image_amp

    def get_image_by_attribute_name(self,image_attr):
        """
        """
        
        itype = self.get_image_attribute_name(image_attr)
        try:
            image = getattr(self,itype).copy()
        except AttributeError:
            return None
            
        if self.execute_process_in_amp is not None:
            image = np.ma.array(image, mask=self.amplifier[self.execute_process_in_amp]['image'])
            for attr in ['mask_image_overscan_cols', 'mask_image_prescan_cols', 'mask_image_active_region','rows','cols','mask_full_extension']:
                setattr(self,attr,self.amplifier[self.execute_process_in_amp][attr])
        else:
            ### It does the same as get_image_attribute_name
            return itype

        return image

    def set_image_by_attribute_name(self,attr,image,**kargs):
        
        if image is not None:
            if not hasattr(self,attr):
                empty = np.empty(image.shape)
                empty[:] = np.nan
                setattr(self,attr,empty.copy())
            
            image_attr = getattr(self,attr)
            image_attr[~image.mask] = image[~image.mask]        
            setattr(self,attr,image_attr)

        for attr in kargs.keys():
            if not hasattr(self,attr):
                setattr(self,attr,{})
            attr_dict = getattr(self,attr)
            attr_dict.update({self.execute_process_in_amp:kargs[attr]})
       
    def set_amplifier(self,amp):
        """Set attribute that controls the amplifier that is being processed (for Leach systems)
        """
        setattr(self,'execute_process_in_amp',amp)

 
    def get_pedestal_substracted_image(self,image,pedestal,attr):
        """
        """
        
        #### BUILD PEDESTAL IMAGE TO BE SUBTRACTED FOR A GIVEN AMPLIFIER 
        # (should be inverted: 0-for masked, to remove 0, 1:-for unmasked to remove ped)
        mask_full_extension = np.logical_not(self.amplifier[self.execute_process_in_amp]['mask_full_extension']).astype(float)

        if not hasattr(self,attr):
            pedestal_image = image.data.copy()
        else:
            pedestal_image = getattr(self,attr)
        for ax in pedestal.keys():
            pedestal_image = pedestal_image - np.nan_to_num(pedestal[ax],nan=0.0)*mask_full_extension

        setattr(self,attr,np.array(pedestal_image.data))
        return pedestal_image
 
    def apply_simple_threshold_cut(self,threshold,itype="mean_compressed_pedestal_subtracted_e",
            skip_rows=[],skip_cols=[]):
        """Apply threshold cut before clusterization
            
            if thr_in_sigma is False:
                All pixels must have a minimum value, i.e. q_ij > threshold

            if thr_in_sigma is True:
                All pixels must have a minimum value of n_sigmas*sigma0, q_ij > n_sig_min*sigma0
                and each cluster must have at least a pixel with charge  q_ij > n_sig_at_least*sigma0

                in this case threshold is a list of three values: (sigma0,n_sig_min, n_sig_at_least) 
            
            threshold in units of eV
        """
        iftype = self.get_image_attribute_name(itype)
        image = self.get_image_by_attribute_name(itype).copy()
        # CLUSTERIZATION IS DONE AMPLIFIER FOR AMPLIGIER
        image[image.mask] = 0.0
        image = image.data
        
        # SET PIXEL CHARGE TO ZERO FOR THOSE REGION THAT SHOULD NOT BE CLUSTERIZED
        # ... pixels outside the active region
        image[self.mask_image_active_region] = 0.0
        # ... any column and row given by the user via skip_rows and skip_cols from ClusterFinderProcess
        _skip_rows = skip_rows[self.execute_process_in_amp] if type(skip_rows)==dict else skip_rows
        for r in _skip_rows:
            if type(r)==list:
                for ri in np.arange(*r):
                    image[ri,:] = 0.0
                image[int(ri+1),:] = 0.0
            else:
                image[int(r),:] = 0.0
        _skip_cols = skip_cols[self.execute_process_in_amp] if type(skip_cols)==dict else skip_cols
        for c in _skip_cols:
            if type(c)==list:
                for ci in np.arange(*c):
                    image[:,int(ci)] = 0.0
                image[:,int(ci+1)] = 0.0
            else:
                image[:,int(c)] = 0.0

        #    APPLY PIXEL CHARGE THRESHOLD TO IGNORE THOSE PIXELS WITH LOWER CHARGE
        # ######################################################################################
        #       threshold = sigma_eV, n_noise , n_seed, std_eV
        #   CHANGE UNITS ACCORDING TO INPUT IMAGE GIVEN IN UNITS OF e- or ADUs
        CF = 1/u.e2eV
        CFunits = "e-"
        if not itype.endswith("_e"):
            CF = CF*self.ADC2e[self.execute_process_in_amp] if hasattr(self,'ADC2e') else CF*u.ADC2e
            CFunits = "ADU"
        q_cut  = threshold[1]*(threshold[0]*CF)
        q_seed = threshold[2]*(threshold[0]*CF)
        self.__attr_to_clusterize__="_threshold_cut" 
        self.y_threshold_cut, self.x_threshold_cut = np.where(image >= q_cut)
        print(f"   - number of pixels that survive the cut {len(self.y_threshold_cut)} [q_cut={q_cut} {CFunits} assuming {u.e2eV} eV/e ]")
        
        # array for each x and y values that should be clusterized
        self.pixel_charge = image[self.y_threshold_cut,self.x_threshold_cut]

        # boolean array to FLAG those pixels that acts as cluster seed --> should be used to speed up clusterization algorithm
        self.pixel_cluster_seed = self.pixel_charge >= q_seed
        
        # add attributs to hits that should be kept in the output root file
        setattr(self,"sigma0", threshold[0])
        setattr(self,"q_min_cls_seed", threshold[0]*threshold[2])

        image[image<=q_cut]=0.0
        self._image_to_find_clusters_ = image.copy()
        
        ### add q_cut and q_seed
        self.q_cut = q_cut
        self.q_seed = q_seed
    
    def get_compatibility_with_ClusterObject(self,run_id=0,image_id=0):
        """

        Event   : int
            corresponds to the run number id
        ccd     : int
            corresponds to the image number

        """
        self.event = int(run_id)
        self.ccd   = int(self.ccd)

        null_array = np.ones_like(getattr(self,"x{}".format(self.__attr_to_clusterize__)))
        for var in ["z"]:
            setattr(self,"{}{}".format(var,self.__attr_to_clusterize__),-1*null_array )
        
        ### XXX clusterization mix this two components!!! XXX
        #row = getattr(self,"y{}".format(self.__attr_to_clusterize__)).copy()
        #col = getattr(self,"x{}".format(self.__attr_to_clusterize__)).copy()
        #setattr(self,"x{}".format(self.__attr_to_clusterize__),row)
        #setattr(self,"y{}".format(self.__attr_to_clusterize__),row)

        ### XXX pixel charge is in units of electrons, convert to eV: use u.e2eV XXX
        setattr(self,"Edep{}".format(self.__attr_to_clusterize__), self.pixel_charge*u.e2eV)
        
        ### MCT if exists
        if hasattr(self,"mct_image"):
            nrows = getattr(self,"y{}".format(self.__attr_to_clusterize__)).copy()
            ncols = getattr(self,"x{}".format(self.__attr_to_clusterize__)).copy()
            setattr(self, "Edep_pixels_MCT", self.mct_image[nrows,ncols])
            setattr(self, "Edep_pixels_MCT_ID", self.mct_cls_id[nrows,ncols])

    def calibrate_signal(self,image):

        return image/u.ADC2e
    
    def imshow_equalized(self,im,bins=None,vmin=None,vmax=None,palette='viridis'):
        
        image = self.get_equalized_image(im)
        if display:
            fig,ax = plt.subplots(1,1,figsize=(15,8))
            _ = ax.imshow( image, cmap=palette, aspect='auto', vmin=vmin, vmax=vmax )
            plt.colorbar(_)
            plt.show()
            return

        return image

    def get_equalized_image(self,image,nbins=None,**kwargs):
        """Equalize image to highlight noise patter, but also signal
        """
        
        # equaliz
        nbins = nbins or int(np.sqrt(image.size))*5
        imhist,bins = np.histogram(image.flatten(),nbins, density=True)
        cdf = imhist.cumsum()
        cdf = cdf / cdf[-1]
        
        im2 = np.interp(image.flatten(),bins[:-1],cdf)
        image = im2.reshape(image.shape)
        
        return image

def BuilderRawData(file_name,file_config=None):
    """
    """
        
    # Reading the format file from its name, to properly instanciate RawData
    file_format = file_name.split(".")

    if len(file_format) == 2:
        # <file_name>.fits
        _type = (file_format[-1]).lower()
    elif len(file_format) == 3:
        # <file_name>.fits.fz also allowed
        _subtype = (file_format[-1]).lower()
        _type    = (file_format[-2]).lower()
    else:
        msm = "Unknown file format"
        raise NotImplementedError(msm)
    
    if file_config is not None and not type(file_config) == dict:
        raise AttributeError("<BuilderRawData>: file_config must be a dictionary (to use the default configuration do not use this constructor)")
    if type(file_config) == dict:
        if len(set(file_config.keys()) & set(config_fits_raw_image['input'].keys()))==0:
            raise AttributeError("<BuilderRawData>: file_config must be a dictionary with at least one of this keys {}".format(config_fits_raw_image['input'].keys()))

    # INSTANCIATE RawClass according to type format
    if _type == "fits" or _type == "fz" or _type == "bz2":
        default_fits_config = config_fits_raw_image['input']
        if file_config is None:
            file_config={}
        final_user_fits_config={}
        for block_key in default_fits_config.keys():
            if block_key in file_config:
                block_config = file_config[block_key]
            else:
                block_config = default_fits_config[block_key]
            ## check all parameters for each block
            for key_in_block in sorted(default_fits_config[block_key].keys()):
                if not key_in_block in block_config.keys():
                    block_config[key_in_block] = default_fits_config[block_key][key_in_block]

            final_user_fits_config[block_key] = block_config
        
        extension = final_user_fits_config['image']['extensions']	

        return FitsRawData(file_name,extension,final_user_fits_config)
    else:
        msm = "file format '{}' is not implemented yet.".format(_type)
        raise NotImplementedError(msm)


###############################################################################################
#####   SPECIFIC CLASSES TO LOAD DIFFERENCT TYPES OF RAW DATA
###############################################################################################
class FitsRawData(RawData):
    """


    Example
    -------	       
       Assume we have the following raw image (7rows x 11cols) in a fits file format
	
	           0    ooooooooooo           
	      r    1    ooooooooooo           label code:  o - region for the overscan
	      o    2    oXXXXXXXooo                        x - active region of the sensor
	      w    3    oXXXXXXXooo
	      s    4    oXXXXXXXooo
	           5    oXXXXXXXooo
	           6    ooooooooooo
	   
	                0123456789..   (<-- columns || axis=0 || NAXIS1)
	                       
	   There are two ways to read properly the active area and the overscan:
		1. by defining the number of rows and colums for the overscan on top, 
                    bottom, left and right of the active area
        	     
        	     rows:
	               n_rows_overscan = 1
        	       n_rows_prescan = 3
	             cols:
        	       n_cols_prescan = 2
	               n_cols_overscan = 1

	          This method use two parameters from the fits header: NAXIS1 and NAXIS2 to 
	          get the total number of pixels in the X and Y directions (cols and rows,
	          respectively).

	        2. by defining the active area: which pixel start and end in both axis 
	            rows (Y) and columns (X)

	               image_slice_sensor_rows = (2,5)
        	       image_slice_sensor_cols = (1,7)
        
        There are 4 overscan regions

	          0    L TTTTTTT RRR           
	          1    L TTTTTTT RRR           label code:  o - region for the overscan
	          2    L XXXXXXX RRR                        x - active region of the sensor
	          3    L XXXXXXX RRR
	          4    L XXXXXXX RRR
	          5    L XXXXXXX RRR
	          6    L BBBBBBB RRR

            T   overscan on the top
            B   overscan on the bottom
            L   overscan on the left
            R   overscan on the right

    """

    def __init__(self,file_name,extension,pconfig):
        super().__init__(file_name)
        
        ### FULL IMAGE RAW DATA:  full raw image: active region, and overscan 
        self.ACM = False
        if pconfig['image']['ACM_multi_extension'] and type(extension)==list:
            self.image = fits.getdata(self.file_name, extension[0]).astype(float)
            self.ACM_amp_rows = self.image.shape[0]
            self.ACM_amp_cols = self.image.shape[1]
            self.__extensions__ = [extension[0]]
            self.ampl = [f"EXT{self.__extensions__[-1]}"]
            self.extname = self.ampl[-1]

            if len(extension)>1:
                for ext in extension[1:]:
                    self.image = np.concatenate((self.image,fits.getdata(self.file_name,ext).astype(float)),axis=0)
                    #_extname = fits.getval(self.file_name,'extname',ext=ext).replace("CCD_","")
                    _extname = f"EXT{ext}"
                    self.extname += "-"+_extname
                    self.__extensions__.append(ext)
                    self.ampl.append( _extname )

            self.extension = int(extension[0])
            self.ACM = True
            self.ACM_amp_names = self.extname.split("-")

            self.n_amp = len(self.ACM_amp_names)

        else:
            self.extension = int(extension)
            self.image = fits.getdata(self.file_name, self.extension).astype(float)

        ### HEADER:  parameters where Sloc Control Parameters are recorded 
        if 'ext_header' in pconfig['image'] and type(pconfig['image']['ext_header']!=list):
            ext_header = pconfig['image']['ext_header']
        else:
            ext_header = self.extension
        if type(ext_header)==list:
            raise IOError("<FitsRawData>: header extension is a list, and should be an integer")
        self.image_header = fits.getheader(self.file_name, ext_header)

        #  check if the image to be processed is a MONTECARLO Truth IMAGE
        #       IN THIS CASE, READ THE HEADER FROM EXTENSION 1
        if 'EXTNAME' in self.image_header.keys():
            if self.image_header['EXTNAME'] == 'MONTECARLOTRUTH':
                self.image_header = fits.getheader(self.file_name, 0) 

        # CONFIGURATION PARAMETERS
        #   add datetime from fits HEADER
        self.get_datetime(pconfig['datetime'])
        #   in fits HEADER, for slow control parameters: T, P, ...
        self.get_from_header(pconfig['scp'])
        #   in fits HEADER, to interpret data: Nskips, Nrows, Ncols, ..
        self.get_from_header(pconfig['convention'])
        #   for data processing (user input options)
        self.get_from_config(pconfig['image'])
        
        
        #### AFTER READ CONVENTION 

        # FOR NOW IMAGE FITS FILE HEADER DO NOT HAS INFORMATION ABOUT THE SLOW CONTROL --- 
        # FIXME XXX RANDOMLY DEFINED XXX FIXME
        self.T = (np.random.normal(130.0,10.0), np.random.normal(0.0,10.0))
        self.P = (np.random.normal(1e-5,1e-6), np.random.normal(0.0,1e-6))

        # BINNING IN ROWS AND COLUMNS
        # to avoid conflicts with previous version of waders, re-define naming of the binning in cols and rows
        if hasattr(self,'npbin'):
            self.bin_row = self.npbin
        else:
            try:
                self.bin_row = self.image_header['NPBIN']
            except KeyError:
                self.bin_row = 1
        if hasattr(self,'nsbin'):
            self.bin_col = self.nsbin
        else:
            try:
                self.bin_col = self.image_header['NSBIN']
            except KeyError:
                self.bin_col = 1
        
        #### 
        if __verbose__:
            self.print_info()
        
        #self.image = self.image_data
        ### Correct From Leach Bug
        if self.correct_leach_bug and not self.ACM:
            print(" main INFO. Correcting from Leach bug ")
            self.image = CorrectLeachBugProcess(self.image, self.ampl)
            self.__process_chain__ += "/CorrectLeachBugProcess"

        ### Invert polarity of the signal
        if self.correct_polarity and not self.ACM:
            print(" main INFO. Inverting polarity  ")
            self.image = InvertPolarity(self.image)
            self.__process_chain__ += "/InvertPolarity"
        
        ### Re-adapt for averaged input images:
        if not self.skip_image:
            ######################################################################################### NON-SKIP IMAGE
            print(f"<panaSKImg> info: Fits file does not contain an skip image, Nskips sets to 1 (from header {self.nskips}).")
            self._NDCM = int(self.image_header[self.nskips_keyword])
            self.image_header[self.nskips_keyword] = 1 
            self.nskips = 1
            self.image_mean_compressed = self.image
        elif self.nskips>1:
            ######################################################################################### FITS FILE WITH SKIPS INCLUDED --> RESHAPE TO 3D
            ### reshape the image according to the number of single skips
            #       Nrows x Ncols x Nskips
            #   - the 3rd axis is the nskips
            #
            ### FIXME re-adapt this for binned images
            self.block_compression = np.array([1,1])
            self.block_compression[self.axis_to_compress] = self.nskips
            nrows, ncols = np.array(self.image.shape/self.block_compression).astype(int)
            self.image = self.image.reshape(nrows, ncols, self.nskips)
            print(f"<panaSKImg> info: Image with skips reshape to 3D: (Nrows x Ncols x Nskips)=({self.image.shape[0]} x {self.image.shape[1]} x {self.image.shape[2]})")
        elif int(self.row_binning)>1:
            ######################################################################################### DO SOFTWARE POSTBINNING
            binning = int(self.row_binning)
            nrows,ncols = self.image.shape
            self.image = self.image.reshape(nrows//binning,binning,ncols).sum(axis=1)
            self.image_mean_compressed = self.image
            self.nrows = self.image_mean_compressed.shape[0]
            print(f"<panaSKImg> info: Image with SOFTWARE POSTBINNING: {self.image.shape[0]}x{self.image[1]}")
    
    def get_histogram_image(self, image, n_sigma=3, bins_min=None):
        """ Create a histogram of an image (or any data set) of a resonalbe range with integer (ADU)
        spaced bins

        """
        med = np.median(image) 
        mad = np.median(np.abs(image-np.median(image)))

        # Create bins +/- 3*mad
        if bins_min and 2.*n_sigma*mad<bins_min:
            bins = np.arange( np.floor(med - bins_min/2), np.ceil(med+bins_min/2) )
        else:
            bins = np.arrange( np.floor(med-n_sigma*mad), np.ceil(med+n_sigma*mad) )

        
        hpix, edges = np.histogram( image, bins=bins, density=True )
        centers = edges[:-1] + np.diff(edges)[0]/2.


        return hpix,centers,edges
    
    def get_datetime(self,header_config):
        """Get Exposure start time, readout start time and readout end time
        """

        self.start = None
        self.start_readout = None
        self.end = None

        if 'exposure_start' in header_config:
            start = self.image_header[header_config['exposure_start']]
            self.start = dateutil.parser.parse(start)

        if 'readout_start' in header_config:
            start_readout = self.image_header[header_config['readout_start']]
            self.start_readout = dateutil.parser.parse(start_readout)

        if 'readout_end' in header_config:
            end = self.image_header[header_config['readout_end']]
            self.end = dateutil.parser.parse(end)

        if 'exposure_time' in header_config:
            # assumed to be in s
            self.exposure_time = self.image_header[header_config["exposure_time"]]*u.s
        else:
            self.exposure_time = 0.0
        print(f"Exposure time: {self.exposure_time}s")

        if 'read_time' in header_config:
            # assumed to be in ms --> to s
            self.read_time = self.image_header[header_config["read_time"]]*u.ms/u.s
        else:
            self.read_time = (self.end - self.start_readout).total_seconds()*u.s
        print(f"Total Read-out time: {self.read_time}s")
        
        ### default values in the fits file is ms ---> move to s
        setattr(self, 'tot_time', (self.exposure_time+self.read_time))
        print(f"Total time (exp+readout): {self.tot_time}s")

    def get_from_header(self,header_config):
        """
        """
        for key_name in header_config.keys():
            try:
                val = self.image_header[header_config[key_name]]
                if key_name.lower() in self.data_member_type:
                    val=self.data_member_type[key_name.lower()]( val )
                setattr(self,key_name.lower(),val)
                setattr(self,key_name.lower()+"_keyword",header_config[key_name])
            except KeyError:
                print("WARNING ERROR: Parameter {} will be ignored, not found in the data image header (extension {})".format(
                    key_name,self.extension))

    def get_from_config(self,pconfig):
        """
        """
        for key_name in pconfig.keys():
            try:
                setattr(self,key_name.lower(),pconfig[key_name])
            except ValueError:
                setattr(self,key_name.lower(), None)
                print("WARNING ERROR: Parameter {} set to 'None' (not found in the data image header)".format(key_name))
            except KeyError:
                print("WARNING ERROR: Parameter {} will be ignored, not found in the data image header (extension {})".format(
                    key_name,self.extension))
        
        if self.skip_image:
            if self.nskips>1:
                if self.image.shape == (self.nrows, self.ncols):
                   ### Header with pixels + skipper
                   self.nallcols = self.ncols
                   self.ncols = int(self.nallcols/self.nskips)
                else:
                   ### Header with only pixels
                   self.nallcols = int(self.ncols*self.nskips)
                   self.ncols = self.ncols
                  
        else:
            self.nallcols = self.ncols
    
    def get_overscan_and_sensor_dimensions_ACM(self):
        """
        """
        ### set correct dimensions in rows
        self.ccd_rows = self.ACM_amp_rows

        # Data from the ACM goes into a single fits file, but store in different extensions
        # FitsRawData.__init__ load the 4 extensions into a single concatenated image along row-axis
        # columns does not change extension by extension, but rows it does: row, 2*row, 3*row, 4*row 
        if self.n_cols_prescan>0 and self.bin_col >  self.n_cols_prescan:
            Dcol_prescan = self.bin_col - self.n_cols_prescan
            self.ccd_cols = self.ccd_cols - Dcol_prescan
            self.n_cols_prescan = self.bin_col
        
        if self.n_rows_prescan>0 and self.bin_row > self.n_rows_prescan:
            Drow_prescan = self.ccd_rows - self.n_rows_prescan
            self.ccd_rows = self.ccd_rows - Drow_prescan
            self.n_rows_prescan = self.bin_row
        
        ##### DEFINING THE OVERSCAN OF A SINGLE EXTENSION
        self.n_cols_overscan = int(self.ncols - self.ccd_cols/float(self.bin_col))
        self.n_cols_prescan = int(self.n_cols_prescan/self.bin_col)
        self.n_rows_prescan = int(self.n_rows_prescan/self.bin_row)



        ##### CREATE THE MASK FOR OVERSCAN, PRESCAN AND ACTIVE REGION FOR EACH AMPLIFIER
        ##################################################################################################################################
        self.amplifier = {}
        for n_ext,amp_name in enumerate(self.ampl):
            #### initialize masks
            self.amplifier[amp_name] = {}
            self.amplifier[amp_name]['image'] = np.ones((self.nrows*len(self.ampl),self.ncols), dtype=bool)

            for mask in ['mask_image_overscan_cols','mask_image_prescan_cols','mask_image_active_region','mask_full_extension']:
                self.amplifier[amp_name][mask] = np.ones((self.nrows*len(self.ampl),self.ncols), dtype=bool)
            ####  OVERSCAN
            if self.n_cols_overscan>0:
                self.amplifier[amp_name]['mask_image_overscan_cols'][n_ext*self.nrows:(n_ext+1)*self.nrows,self.ncols-self.n_cols_overscan:self.ncols]=False
            #### PRESCAN
            if self.n_cols_prescan>0:
                self.amplifier[amp_name]['mask_image_prescan_cols'][n_ext*self.nrows:(n_ext+1)*self.nrows,0:self.n_cols_prescan]=False
            #### ACTIVE REGION
            slice_rows = slice(self.n_rows_prescan+(n_ext*self.nrows),self.nrows-self.n_rows_overscan+(n_ext*self.nrows))
            slice_cols = slice(self.n_cols_prescan,self.ncols-self.n_cols_overscan)
            self.amplifier[amp_name]['mask_image_active_region'][slice_rows,slice_cols]=False
            #### EXTENSION MASK
            self.amplifier[amp_name]['mask_full_extension'][n_ext*self.nrows:(n_ext+1)*self.nrows,:]=False
            #### IMAGE MASK
            self.amplifier[amp_name]['image'][n_ext*self.nrows:(n_ext+1)*self.nrows,:]=False
           
            ### ADD rows and cols
            self.amplifier[amp_name]['amp_image_shape'] = {'rows':(slice_rows.start,slice_rows.stop), 'cols':(0,-1)}
            self.amplifier[amp_name]['rows'] = (slice_rows.start,slice_rows.stop)
            self.amplifier[amp_name]['cols'] = (slice_cols.start,slice_cols.stop)
            
            #from matplotlib import pyplot as plt
            #plt.figure(1)
            #plt.imshow(self.amplifier[amp_name]['mask_image_overscan_cols'], aspect='auto',origin='lower')
            #plt.show(block=True)

        # add atributes to the rawdata object and set them to be None
        # this will be set to proper values during execute process
        for attr in ['mask_image_prescan_cols','mask_image_overscan_cols','mask_image_active_region']:
            setattr(self,attr,None)
        
        # add full active region independetly of amplifier, and add full overscan independently of
        # ampl
        for attr in ['mask_image_overscan_cols','mask_image_prescan_cols','mask_image_active_region']:
            masks = []
            for amp in self.amplifier.keys():
                masks.append(self.amplifier[amp][attr])
            # sum all masks
            full_mask = masks[0]
            for m in masks[1:]:
                full_mask = np.logical_or(full_mask,~m)
            setattr(self,attr.replace('_image_','_'),~full_mask)

        if self.n_cols_overscan < 0:
            raise IOError("<IOERROR,FitsRawData>: NUMBER OF COLUMNS IN THE OVERSCAN IS NEGATIVE!!!!!!!! ")

    def get_overscan_and_sensor_dimensions(self):
        """
        The sensitive region is defined by the physical number of pixels that has been build during
        CCD production and should be defined in the JSON file using the values

            --------------
            |     |      |
            |     |      |
            |     |      |
            --------------
      L2[-------------------]U2

      In this sketh, we can see how a CCD geometry looks like, we have the active region matrix with
      n_cols x n_rows pixels, that are defined by the manufacter, and then we have the serial
      register, that has a given number of pre-scan columns n_cols_prescan (the same for each
      amplifier).

      So, with these three parameters and knowing the number of amplifiers used to read a given
      number of rows (all columns will always be read out) the overscan can be estimated as follow

        
        
        n_cols_overscan = (X_size - n_cols)/n_amp - n_cols_prescan


        """
        if self.ACM:
            self.get_overscan_and_sensor_dimensions_ACM()
            return

        # self.nrows and self.ncols is the real dimensions of the CCD without accounting for the
        # number of skips!!!!
        if not hasattr(self,'n_amp'):
            self.n_amp = 2 if self.ampl=="UL" else 1
        
        ### GET THE SIZE OF THE OVERSCAN
        if int(self.n_amp)==1:
            # when data is binning, we must redefine the prescan number of cols and rows
            if self.n_cols_prescan>0 and self.bin_col >  self.n_cols_prescan:
                Dcol_prescan = self.bin_col - self.n_cols_prescan
                self.ccd_cols = self.ccd_cols - Dcol_prescan
                self.n_cols_prescan = self.bin_col
            if self.n_rows_prescan>0 and self.bin_row > self.n_rows_prescan:
                Drow_prescan = self.ccd_rows - self.n_rows_prescan
                self.ccd_rows = self.ccd_rows - Drow_prescan
                self.n_rows_prescan = self.bin_row

            ##### DEFINING THE OVERSCAN OF AN IMAGE WITH ONE AMPLIFIER
            # self.n_cols_overscan = int((self.ncols - self.ccd_cols/float(self.bin_col)) - self.n_cols_prescan/float(self.bin_col))
            # ccd_cols : pre + active
            self.n_cols_overscan = int(self.ncols - self.ccd_cols/float(self.bin_col))

            # get mask for the different regions
            # apply binning to the number of cols/rows
            self.n_cols_prescan = int(self.n_cols_prescan/self.bin_col)
            self.n_rows_prescan = int(self.n_rows_prescan/self.bin_row)
        
        elif int(self.n_amp)==2:
            ##  FIXME  IS ALSO NEEDED FOR 1-AMP ???? 
            # PRESCAN COLUMNS
            if self.n_cols_prescan>0 and self.bin_col >  self.n_cols_prescan:
                # When the number of pixels in the prescan is smaller than the used binning, set the
                # first 'binned column' as prescan as it will contain  pixels from the prescan but
                # also pixels from the active region ... 
                # --- number of pixels from active region that will be in the first binned pixel
                Dcol_prescan = self.bin_col - self.n_cols_prescan
                # --- remove the previous number of pixels that goes to the first binned pixel
                #       from the total number of pixels in the active region
                self.ccd_cols = self.ccd_cols - Dcol_prescan
                # --- now, the prescan columns is one binned pixel, i.e. the binning on cols
                self.n_cols_prescan = self.bin_col
               

            if self.n_rows_prescan>0 and self.bin_row > self.n_rows_prescan:
                # same as in cols, if the binning is smaller in pixel size, take the first binned
                # row as prescan
                Drow_prescan = self.ccd_rows - self.n_rows_prescan
                self.ccd_rows = self.ccd_rows - Drow_prescan
                self.n_rows_prescan = self.bin_row

            # assuming ampifier U and L are read (as they are the only skipper amplifiers)
            
            ### binned CCD dimensions, and pedestal
            self.ccd_cols = self.ccd_cols/float(self.bin_col)
            self.ccd_rows = self.ccd_rows/float(self.bin_row)
            self.n_cols_prescan = int(self.n_cols_prescan/float(self.bin_col))

            ### get the overscan: data dimension from fits file - prescan - ccd_cols
            self.n_cols_overscan = int(np.ceil((self.ncols - 2.0*self.n_cols_prescan - self.ccd_cols)/float(self.n_amp)))

            # XXX in the U region, the skips will be stored in the inverted order
            # IF the image is read by two amplifiers L and U, the NDCM in the U-region should be inverted
            # where || highlight the columns read by U-region XXX
            if self.nskips>1:
                print("Flip skip measurement only on the U-side  ... ")
                self.image_flip = self.image.copy()
                # flip the axis of the NDCM only on the U-side
                Uside_col_start = self.image.shape[1]//2
                self.image[:,Uside_col_start:,:] = np.flip(self.image[:,Uside_col_start:,], axis=2)
        else:
            # not defined yet!
            raise IOError("Number of amplifiers must be 1 or 2 (4 is not yet implemented).")
        
        ##### CREATE THE MASK FOR OVERSCAN, PRESCAN AND ACTIVE REGION FOR EACH AMPLIFIER
        ##################################################################################################################################
        self.amplifier = {}
        if int(self.n_amp)==1:
            print(" -- Readout done with one amplifier ")
            ### XXX FIXME once the name of the amplifier appears in the fits file header
            
            if hasattr(self,'amp_name') and self.amp_name.count('EXT')>0:
                amp_name = self.amp_name
            else:
                amp_name =  f"EXT{self.extension}"
                setattr(self,"extname",amp_name)

            #### initialize masks
            self.amplifier[amp_name] = {}
            self.amplifier[amp_name]['image'] = np.ones((self.nrows,self.ncols), dtype=bool)

            for mask in ['mask_image_overscan_cols','mask_image_prescan_cols','mask_image_active_region','mask_full_extension']:
                self.amplifier[amp_name][mask] = np.ones((self.nrows,self.ncols), dtype=bool)
            ####  OVERSCAN
            if self.n_cols_overscan>0:
                self.amplifier[amp_name]['mask_image_overscan_cols'][:,self.ncols-self.n_cols_overscan:self.ncols]=False
            #### PRESCAN
            if self.n_cols_prescan>0:
                self.amplifier[amp_name]['mask_image_prescan_cols'][:,0:self.n_cols_prescan]=False
            #### ACTIVE REGION
            slice_rows = slice(self.n_rows_prescan,self.nrows-self.n_rows_overscan)
            slice_cols = slice(self.n_cols_prescan,self.ncols-self.n_cols_overscan)
            self.amplifier[amp_name]['mask_image_active_region'][slice_rows,slice_cols]=False
 
            ### ADD rows and cols
            self.amplifier[amp_name]['rows'] = (slice_rows.start,slice_rows.stop)
            self.amplifier[amp_name]['cols'] = (slice_cols.start,slice_cols.stop)
       
            #### EXTENSION MASK
            self.amplifier[amp_name]['mask_full_extension'][:,:]=False
            #### Image mask
            self.amplifier[amp_name]['image'][:,:]=False


        elif int(self.n_amp)==2:  
            print(" -- Readout done with two amplifier ")
            #### initialize masks
            for amp in ['U','L']:
                self.amplifier[amp] = {}
                self.amplifier[amp]['image'] = np.ones((self.nrows,self.ncols), dtype=bool)
                for mask in ['mask_image_overscan_cols','mask_image_prescan_cols','mask_image_active_region','mask_full_extension']:
                    self.amplifier[amp][mask] = np.ones((self.nrows,self.ncols), dtype=bool)
            
            #### for each amplifier add the image mask, to select the part of the 2D array read for each amplifier
            self.amplifier['L']['image'][:,:int(self.ncols/2)]=False
            self.amplifier['U']['image'][:,int(self.ncols/2):]=False

            ####  OVERSCAN(s)
            self.amplifier['L']['mask_image_overscan_cols'][:,slice(int(self.ncols/2-self.n_cols_overscan),int(self.ncols/2))]=False
            self.amplifier['U']['mask_image_overscan_cols'][:,slice(int(self.ncols/2),int(self.ncols/2+self.n_cols_overscan))]=False
            ####  PRESCAN(s)
            self.amplifier['L']['mask_image_prescan_cols'][:,slice(0,self.n_cols_prescan)]=False
            self.amplifier['U']['mask_image_prescan_cols'][:,slice(self.ncols-self.n_cols_prescan,self.ncols)]=False
            #### ACTIVE REGION
            slice_rows = slice(self.n_rows_prescan,self.nrows-self.n_rows_overscan)
            slice_cols = slice(self.n_cols_prescan,int(self.ncols/2-self.n_cols_overscan))
            self.amplifier['L']['mask_image_active_region'][slice_rows,slice_cols]=False
            self.amplifier['L']['rows'] = (slice_rows.start,slice_rows.stop)
            self.amplifier['L']['cols'] = (slice_cols.start,slice_cols.stop)

            #### ACTIVE REGION
            slice_cols = slice(int(self.ncols/2+self.n_cols_overscan),self.ncols-self.n_cols_prescan)
            self.amplifier['U']['mask_image_active_region'][slice_rows,slice_cols]=False
            self.amplifier['U']['rows'] = (slice_rows.start,slice_rows.stop)
            self.amplifier['U']['cols'] = (slice_cols.start,slice_cols.stop)
            #### EXTENSION MASK
            self.amplifier['L']['mask_full_extension'][:,:self.ncols//2]=False
            self.amplifier['U']['mask_full_extension'][:,self.ncols//2:]=False
            #### IMAGE MASK
            self.amplifier['L']['image'][:,:self.ncols//2]=False
            self.amplifier['U']['image'][:,self.ncols//2:]=False

        # add atributes to the rawdata object and set them to be None
        # this will be set to proper values during execute process
        for attr in ['mask_image_prescan_cols','mask_image_overscan_cols','mask_image_active_region']:
            setattr(self,attr,None)
        
        # add full active region independetly of amplifier, and add full overscan independently of
        # ampl
        for attr in ['mask_image_overscan_cols','mask_image_prescan_cols','mask_image_active_region']:
            masks = []
            for amp in self.amplifier.keys():
                masks.append(self.amplifier[amp][attr])
            # sum all masks
            full_mask = masks[0]
            for m in masks[1:]:
                full_mask = np.logical_or(full_mask,~m)
            setattr(self,attr.replace('_image_','_'),~full_mask)

        
        if self.n_cols_overscan < 0:
            raise IOError("<IOERROR,FitsRawData>: NUMBER OF COLUMNS IN THE OVERSCAN IS NEGATIVE!!!!!!!! ")


    def print_info_ccd_region(self):
         
        print("  INFO: Rows and column region for the overscan, prescan and Active regions:")
        for amp in sorted(self.amplifier.keys()):
            for m in sorted(self.amplifier[amp].keys()):
                if m in ['rows','cols','amp_image_shape']:
                    continue
                indxs = np.where(~self.amplifier[amp][m])
                try:
                    rS,cS = indxs[0][0],indxs[1][0]
                    rE,cE = indxs[0][-1],indxs[1][-1]
                except IndexError:
                    continue
                print("     amplifier {} - {}: \t {} rows [{},{}] and {} cols [{},{}]".format(amp,m,int(rE-rS)+1,rS,rE,int(cE-cS)+1,cS,cE))

            print("")
    
    def print_info(self):
        """
        """

        print("Print INFO.")
        print("General info of the fits file")
        print("****************************************************************************")
        print(fits.info(self.file_name))
        print("Displaying full header ext=",self.extension)
        print("****************************************************************************")
        print(repr(self.image_header))
        print("****************************************************************************")
        print("Summary: \n")
        print("N. rows={} columns={} skips={} ".format(self.nrows, self.ncols, self.nskips))
        print("Image ndim: ", self.image.ndim)
        print("Image shape:", self.image.shape) 
        print("Image size: ", self.image.size)    
        print("Image dtype:", self.image.dtype)
        print("****************************************************************************")


    def SaveAsFits(self,output,image_names,naming=None,overwrite=True,bz2=False):
        if len(image_names)==0:
            return
        
        # Define which extensions must be stored as integer masks.
        MASK_NAMES = {"FULLMASK", "MASK", "CLSMASK"}

        
        def _prepare_data(name):
            """Extract plain ndarray data from MaskedArray and cast it to the desired on-disk FITS dtype
            """
            arr = getattr(self, name)
            raw = np.ma.getdata(arr)
            if name.upper() in MASK_NAMES:
                return np.asarray(raw, dtype=np.uint8)
            return np.asarray(raw, dtype=np.float32)
        
        if naming is not None and len(naming) != len(image_names):
            raise ValueError("naming must have the same length as image_names")

        header = self.image_header
        for pkey in u._tohdr.keys():
            val, txt = u._tohdr[pkey]
            header[pkey.replace('EXT','')] = (val,txt)        
        header['FILE']   = self._file_name
        try:
            header['FILEEXT']= ','.join(map(str,self.__extensions__))
        except AttributeError:
            header['FILEEXT'] = str(self.extension)

        # Primary HDU
        primary_data = _prepare_data(image_names[0])
        hdu_list = [fits.PrimaryHDU(data=primary_data, header=header)]
        if naming is not None:
            hdu_list[-1].name = naming[0]
        
        # Extract non-primary image extensions
        if len(image_names)>1:
            for k, name in enumerate(image_names[1:], start=1):
                data = _prepare_data(name)
                hdu = fits.ImageHDU(data=data)
                if naming is not None:
                    hdu.name = naming[k]
                hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        if bz2 and not output.endswith(".bz2"):
            if output.endswith(".fits"):
                output = output + ".bz2"
            else:
                output = output + ".fits.bz2"

        hdul.writeto(output, overwrite=overwrite)
    
        return

