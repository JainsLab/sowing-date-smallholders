import ee
ee.Initialize()

def get_collection(start_date, end_date, roi, collection='COPERNICUS/S2_SR_HARMONIZED'):
    """ Filters an ee.ImageCollection.

        Args:
            start_date (str): start date of the filter (YYYY-MM-DD)
            end_date (str): end date of the filter (YYYY-MM-DD)
            roi (ee.geometry, ee.FeatureCollection): the geometry to intersect
            collection (str): the desired dataset (https://developers.google.com/earth-engine/datasets)

        Returns:
            An ee.ImageCollection
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    filtered_collection = ee.ImageCollection(collection) \
        .filterBounds(roi) \
        .filterDate(start, end)
    return filtered_collection




def hls_30m_daily(path_ee, path_aux, start_date, end_date, roi):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    hlss = ee.ImageCollection(path_ee) \
        .filterBounds(roi) \
        .filterDate(start, end)
    hlsl = ee.ImageCollection(path_aux) \
        .filterBounds(roi) \
        .filterDate(start, end)
    
    return hlss, hlsl

def modis_250m_daily(asset_modis_250m, asset_modis_500m, start_date, end_date, geometry):
    
    modisBands_250m = ['sur_refl_b01', 'sur_refl_b02']
    lsBands_250m = ['red', 'nir']

    # Helper function to extract the QA bits
    def get_qa_bits(image, start, end, new_name):
        # Compute the bits we need to extract.
        pattern = 0
        for i in range(start, end + 1):
            pattern += 2**i

        # Return a single band image of the extracted QA bits, giving the band a new name.
        return image.select([0], [new_name]).bitwiseAnd(pattern).rightShift(start)

    # Function to mask out cloudy pixels.
    def mask_quality(image):
        # Select the QA band.
        QA = image.select('state_1km')
        # Get the internal_cloud_algorithm_flag bit.
        internal_quality = get_qa_bits(QA, 8, 13, 'internal_quality_flag')
        # Return an image masking out cloudy areas.
        return image.updateMask(internal_quality.eq(0))

    # Function to mask out cloudy pixels.
    def mask_quality_250m(image):
        # Select the QA band.
        QA = ee.Image(image.get('cloud_mask')).select('state_1km')
        # Get the internal_cloud_algorithm_flag bit.
        internal_quality = get_qa_bits(QA, 8, 13, 'internal_quality_flag')
        # Return an image masking out cloudy areas.
        return image.updateMask(internal_quality.eq(0))

    modis_250m = (ee.ImageCollection(asset_modis_250m)
                  .filter(ee.Filter.date(start_date, end_date)))

    modis_500m = (ee.ImageCollection(asset_modis_500m)
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date))

    modis_250m_clouds = (ee.Join.saveFirst('cloud_mask')
                         .apply(primary=modis_250m,
                                secondary=modis_500m.select("state_1km"),
                                condition=ee.Filter.equals(
                                    leftField='system:index',
                                    rightField='system:index')))

    modis_250m_clouds = (ee.ImageCollection(modis_250m_clouds)
                         .map(mask_quality_250m)
                         .select(modisBands_250m, lsBands_250m))

    return modis_250m_clouds


def add_mask_band_s2(image):
    mask = image.select('B8').mask()  # Get the mask of the image (1 for valid pixels, 0 for masked)
    mask = mask.rename("mask")  # rebane
    return image.addBands(mask)
def add_mask_band_modis(image):
    mask = image.select('nir').mask()  # Get the mask of the image (1 for valid pixels, 0 for masked)
    mask = mask.rename("mask")  # rebane
    return image.addBands(mask)

def add_mask_band_hls(image):
    mask = image.select('NIR').mask()  # Get the mask of the image (1 for valid pixels, 0 for masked)
    mask = mask.rename("mask")  # rebane
    return image.addBands(mask)


def calc_ndvi_s2(image):
    """ Calculates the NDVI (Normalized Difference Vegetation Index).

        Args:
            image: ee.Image

        Returns:
            Adds the NDVI image to the ee.Image
    """
    
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# ------------------------------------------------------------------
# Helper to add 30‑m NDVI and make every band 30‑m
# ------------------------------------------------------------------
def calc_ndvi_s2_reproject(image):
    """
    Adds an NDVI band (B8 – B4) and reprojects all bands to 30 m.

    Args
    ----
    image : ee.Image
        A Sentinel‑2 L2A image.

    Returns
    -------
    ee.Image
        The input image + NDVI band, with every band set to 30 m
        (bilinear up‑sampling from native 10 m / 20 m resolutions).
    """
    # Select and reproject red and nir bands to 30 m
    red = image.select('B4').reproject(crs=image.select('B4').projection(), scale=30)
    nir = image.select('B8').reproject(crs=image.select('B8').projection(), scale=30)



    # Compute NDVI using reprojected bands
    ndvi = (nir.subtract(red)).divide(nir.add(red)).rename('NDVI')


    # # 3) Choose a reference projection (all S‑2 bands share CRS;
    # #    we just need one of them, e.g. B4)
    # ref_proj = image.select('B8').projection()   # same CRS, 10 m nominal

    # # 4) Resample to avoid jagged nearest‑neighbour artefacts when up‑scaling,
    # #    then reproject so the default scale is 30 m
    # ndvi = ndvi.reproject(crs=ref_proj, scale=30)
    

    return image.addBands(ndvi)

def calc_ndvi_modis(image):
    """ Calculates the NDVI (Normalized Difference Vegetation Index).

        Args:
            image: ee.Image

        Returns:
            Adds the NDVI image to the ee.Image
    """
    ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI')
    return image.addBands(ndvi)

def calc_ndvi_hls(image):
    """ Calculates the NDVI (Normalized Difference Vegetation Index).

        Args:
            image: ee.Image

        Returns:
            Adds the NDVI image to the ee.Image
    """
    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')
    return image.addBands(ndvi)


def mask_fmask(image):
    """ Applies the Fmask cloud mask. HLS data
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002#bands
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002#bands
    """
    fmask = image.select('Fmask')
    
    # Define the Fmask bitwise operators to isolate cloud, cloud shadow, snow/ice
    cloud_bit_mask = 1 << 1  # Bit 1: Cloud
    shadow_bit_mask = 1 << 3  # Bit 3: Cloud shadow
    snow_bit_mask = 1 << 4  # Bit 4: Snow/ice

    # Combine conditions of unwanted features being present using bitwise operations
    mask = (fmask.bitwiseAnd(cloud_bit_mask)
        .Or(fmask.bitwiseAnd(shadow_bit_mask))
        .Or(fmask.bitwiseAnd(snow_bit_mask))
        .eq(0))  # Keep if none are present

    return image.updateMask(mask)

def cloud_mask_qa(image):
    """ Applies the S2 QA mask.

Args:
    image (ee.Image): ee.Image

Returns:
    Updates the cloud mask in the ee.Image
    """
    cloudShadowBitMask = (1 << 10)
    cloudsBitMask = (1 << 11)
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)


def cloud_mask_probability(image, max_cloud_prob=65):
    """ Applies the S2 cloud probability mask.

        Args:
            image (ee.Image): ee.Image
            max_cloud_prob: The cloud probability threshold

        Returns:
            Updates the cloud mask in the ee.Image
    """
    clouds = ee.Image(image.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(max_cloud_prob)
    return image.updateMask(isNotCloud)


def mask_edges(image):
    """ Sometimes, the mask for the 10m bands does not exclude bad pixels at the image edges.
        Therefore, it's necessary to apply the masks for the 20m and 60m bands as well.
        Reference: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY#description

        Args:
            image (ee.Image): ee.Image to which the mask is to be applied

        Returns:
            Updates the cloud mask in the ee.Image

    """
    return image.updateMask(image.select('B8A').mask().updateMask(image.select('B9').mask()))


def create_reduce_region_function(geometry,
                                  reducer=ee.Reducer.mean(),
                                  scale=10,
                                  crs='EPSG:3857',
                                  bestEffort=True,
                                  maxPixels=1e13,
                                  tileScale=4):
    def reduce_region_function(img):
        """Applies the ee.Image.reduceRegion() method.
            Reference: https://developers.google.com/earth-engine/tutorials/community/time-series-visualization-with-altair
            Args:
                img:
                    An ee.Image to reduce to a statistic by region.

            Returns:
                An ee.Feature that contains properties representing the image region
                reduction results per band and the image timestamp formatted as
                milliseconds from Unix epoch (included to enable time series plotting)
        """
        try:
            stat = img.reduceRegion(
                reducer=reducer,
                geometry=geometry,#.transform(crs, 0.001)
                scale=scale,
                crs=crs)
                # bestEffort=bestEffort,
                # maxPixels=maxPixels,
                # tileScale=tileScale)

            return ee.Feature(geometry, stat).set({ 
                'millis':img.date().millis(),#img.date().millis()   'millis': img.get('system:time_start'),
                'id': img.id()
                }).setGeometry(None)
        except ee.EEException as e:
            # Handle the exception (e.g., print an error message)
            print(f"Error processing image: {e}")
            return None

    return reduce_region_function

def create_pixel_count_function(geometry, scale=10):
    def pixel_count_function(img):
        # Ensure 'scale' is an ee.Number if derived from Earth Engine methods
        scale_ee_number = ee.Number(scale)

        # Buffering the geometry using a percentage of the scale
        buffered_geometry = geometry.buffer(distance=scale_ee_number.divide(2), maxError=scale_ee_number.divide(10))

        """Counts total, masked, and unmasked pixels in the 'mask' band."""
        total_pixels = img.select('mask').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=buffered_geometry, 
            scale=scale
        ).get('mask')

        masked_pixels =  img.updateMask(img.select('mask').eq(0)).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=buffered_geometry,
            scale=scale
        ).get('mask')

        unmasked_pixels = img.updateMask(img.select('mask').eq(1)).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=buffered_geometry,
            scale=scale
        ).get('mask')
        

        # Calculate total area
        total_area = img.select('mask').gt(-1).multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale
        ).get('mask')

        # Calculate masked area
        mask_zeros = img.select('mask').eq(0)
        masked_area_image = mask_zeros.multiply(ee.Image.pixelArea())
        masked_area = masked_area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale
        ).get('mask')

        # Calculate unmasked area
        unmasked_area = img.updateMask(img.select('mask').eq(1)).multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale
        ).get('mask')

        # return ee.Feature(geometry, {
        #     'total_area': total_area,
        #     'masked_area': masked_area,
        #     'unmasked_area': unmasked_area,
        #     # 'id': img.id()
        # }).setGeometry(None)
        
        return ee.Feature(geometry, {
            'total_pixels': total_pixels,
            'masked_pixels': masked_pixels,
            'unmasked_pixels': unmasked_pixels,
            'total_area': total_area,
            'masked_area': masked_area,
            'unmasked_area': unmasked_area,
            # 'id': img.id()
        }).setGeometry(None)

    return pixel_count_function


def fc_to_dict(fc):
    """ Transfers the properties of the feature to a dictionary.
        Reference: https://developers.google.com/earth-engine/tutorials/community/time-series-visualization-with-altair

        Args:
            fc (ee.FeatureCollection): ee.FeatureCollection

        Returns:
            A dictionary
    """
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()),
        selectors=prop_names).get('list')

    return ee.Dictionary.fromLists(prop_names, prop_lists)
