from bidict import bidict

# Relative path for intermediate results
PATH_IMAGES = "./Results Intermediate/"

# Splitting the dataset
RATIOS = [0.7, 0.15, 0.15]
YEAR_START_TRAIN = 1959 # The split of the US full dataset based on the ratio above
YEAR_START_VAL = 2002  # The split of the US full dataset based on the ratio above
YEAR_START_TEST = 2011  # The split of the US full dataset based on the ratio above
SEQ_LEN = 10
FORECAST_HORIZON = 6
IS_LABEL_SCALED = True

# Constructing batch
# BATCH_SIZE = 32
BATCH_SIZE = 378

# Constructing network 
METRIC = "mse"
FEATURES_M_NAME = "Xm"
FEATURES_LABEL_NAME = "label"
FEATURES_AGE_NAME = "age"
SEED_TORCH = 77

COL_CATEGORIES = ['sex', 'cause']
COL_CATEGORIES_FCNN = ['sex', 'cause', FEATURES_AGE_NAME]
COL_TYPE = 'type'
COL_CO_YR_SX_CA = ['country', 'year', 'sex', 'cause']
COL_CO_YR_SX = ['country', 'year', 'sex']
TYPE_TRUE = 'true'
TYPE_VAL = 'val'
TYPE_TEST = 'test'
TYPE_TRAIN = 'train'

# Minimum age considered
AGE_START = 20

LINESTYLES = [(0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 5)),
              (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)),
              ]

# Mapping for categorical variables

BIDICT_SEX_0_1 = bidict({0: 'male', 1: 'female', 2: "total"})
BIDICT_SEX_1_2 = bidict({1: 'male', 2: 'female', 3: "total"})

BIDICT_CAUSE = bidict({6: 'all-cause',
                       0: 'circulatory',
                       1: 'neoplasm',
                       2: 'respiratory',
                       3: 'digestive',
                       4: 'external',
                       5: 'other'})
BIDICT_CAUSE_1 = bidict({6: 'all-cause',
                         0: 'circulatory diseases',
                         1: 'neoplasms',
                         2: 'respiratory diseases',
                         3: 'digestive system',
                         4: 'external causes',
                         5: 'other'})
BIDICT_CAUSE_1_HMD = bidict({0: 'all-cause',
                             1: 'circulatory diseases',
                             2: 'neoplasms',
                             3: 'respiratory diseases',
                             4: 'digestive system',
                             5: 'external causes',
                             6: 'other'})

BIDICT_AGE_HMD_COD = bidict({f"m{i}": f"{i}" for i in range(1, 101)})

BIDICT_AGE_HMD = bidict({'m0': 0, 'm1': 1, 'm5': 2, 'm10': 3, 'm15': 4, 'm20': 5, 
                         'm25': 6, 'm30': 7, 'm35': 8, 'm40': 9, 'm45': 10, 'm50': 11, 
                         'm55': 12, 'm60': 13, 'm65': 14, 'm70': 15, 'm75': 16, 'm80': 17, 
                         'm85': 18, 'm90': 19, 'm95': 20, 'm100p': 21})
BIDICT_CAUSE_2 = bidict({0: 'ischaemic heart disease', 1: 'cvd and stroke', 2: 'other circulatory system diseases',
                         3: 'bowel cancer', 4: 'liver cancer', 5: 'lung cancer', 6: 'breast cancer', 7: 'prostate cancer', 8: 'other cancers', 9: 'other digestive organ cancers',
                         10: 'influenza and pneumonia', 11: 'chronic lower respiratory disease', 12: 'other respiratory diseases', 
                         13: 'gastric and duodenal ulcer',  14: 'chronic liver disease',  15: 'other digestive system diseases',
                         16: 'traffic accidents', 17: 'self-harm and interpersonal violence', 18: 'other external causes',
                         19: 'aids and tuberculosis',  20: 'diabetes and obesity', 21: 'alcohol abuse and drug dependence', 22: "alzheimer's disease", 23: 'dementia and other mental disorders', 24: 'rest of causes',})

COL_MORTALITY_HMD = ['m0', 'm1', 'm5', 'm10', 'm15', 'm20', 'm25', 
                 'm30', 'm35', 'm40', 'm45', 'm50', 'm55', 'm60', 
                 'm65', 'm70', 'm75', 'm80', 'm85', 'm90', 'm95', 'm100p']
COL_MORTALITY_HMD_MAP = {'m0': '0', 'm1': '1', 'm5': '2', 'm10': '3', 'm15': '4', 'm20': '5', 
                         'm25': '6', 'm30': '7', 'm35': '8', 'm40': '9', 'm45': '10', 'm50': '11', 
                         'm55': '12', 'm60': '13', 'm65': '14', 'm70': '15', 'm75': '16', 'm80': '17', 
                         'm85': '18', 'm90': '19', 'm95': '20', 'm100p': '21'}
# COL_MORTALITY_HMD_MAP = {'m0': '0', 'm1': '1', 'm5': '5', 'm10': '10', 'm15': '15', 'm20': '20', 
#                          'm25': '25', 'm30': '30', 'm35': '35', 'm40': '40', 'm45': '45', 'm50': '50', 
#                          'm55': '55', 'm60': '60', 'm65': '65', 'm70': '70', 'm75': '75', 'm80': '80', 
#                          'm85': '85', 'm90': '90', 'm95': '95', 'm100p': '100'}

# Years of ICD change
YEAR_SEPARATORS_US = [1968, 1979, 1999]