{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Q8X8xDwS6vgS"
   },
   "source": [
    "# Breast Cancer Prediction "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "PEGG4I6n60cM"
   },
   "source": [
    "## Importing the libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "D4b029YC7C-y"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.svm import SVC\n",
    "import time\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Aa536pRY7Eq5"
   },
   "source": [
    "## Importing the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "4POXlqg47Ny3"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sample code number</th>\n",
       "      <th>Clump Thickness</th>\n",
       "      <th>Uniformity of Cell Size</th>\n",
       "      <th>Uniformity of Cell Shape</th>\n",
       "      <th>Marginal Adhesion</th>\n",
       "      <th>Single Epithelial Cell Size</th>\n",
       "      <th>Bare Nuclei</th>\n",
       "      <th>Bland Chromatin</th>\n",
       "      <th>Normal Nucleoli</th>\n",
       "      <th>Mitoses</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000025</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1002945</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1015425</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1016277</td>\n",
       "      <td>6</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1017023</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sample code number  Clump Thickness  Uniformity of Cell Size  \\\n",
       "0             1000025                5                        1   \n",
       "1             1002945                5                        4   \n",
       "2             1015425                3                        1   \n",
       "3             1016277                6                        8   \n",
       "4             1017023                4                        1   \n",
       "\n",
       "   Uniformity of Cell Shape  Marginal Adhesion  Single Epithelial Cell Size  \\\n",
       "0                         1                  1                            2   \n",
       "1                         4                  5                            7   \n",
       "2                         1                  1                            2   \n",
       "3                         8                  1                            3   \n",
       "4                         1                  3                            2   \n",
       "\n",
       "   Bare Nuclei  Bland Chromatin  Normal Nucleoli  Mitoses  Class  \n",
       "0            1                3                1        1      2  \n",
       "1           10                3                2        1      2  \n",
       "2            2                3                1        1      2  \n",
       "3            4                3                7        1      2  \n",
       "4            1                3                1        1      2  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv('breast_cancer.csv')\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 683 entries, 0 to 682\n",
      "Data columns (total 11 columns):\n",
      " #   Column                       Non-Null Count  Dtype\n",
      "---  ------                       --------------  -----\n",
      " 0   Sample code number           683 non-null    int64\n",
      " 1   Clump Thickness              683 non-null    int64\n",
      " 2   Uniformity of Cell Size      683 non-null    int64\n",
      " 3   Uniformity of Cell Shape     683 non-null    int64\n",
      " 4   Marginal Adhesion            683 non-null    int64\n",
      " 5   Single Epithelial Cell Size  683 non-null    int64\n",
      " 6   Bare Nuclei                  683 non-null    int64\n",
      " 7   Bland Chromatin              683 non-null    int64\n",
      " 8   Normal Nucleoli              683 non-null    int64\n",
      " 9   Mitoses                      683 non-null    int64\n",
      " 10  Class                        683 non-null    int64\n",
      "dtypes: int64(11)\n",
      "memory usage: 58.8 KB\n"
     ]
    }
   ],
   "source": [
    "dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sample code number</th>\n",
       "      <th>Clump Thickness</th>\n",
       "      <th>Uniformity of Cell Size</th>\n",
       "      <th>Uniformity of Cell Shape</th>\n",
       "      <th>Marginal Adhesion</th>\n",
       "      <th>Single Epithelial Cell Size</th>\n",
       "      <th>Bare Nuclei</th>\n",
       "      <th>Bland Chromatin</th>\n",
       "      <th>Normal Nucleoli</th>\n",
       "      <th>Mitoses</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>6.830000e+02</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "      <td>683.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.076720e+06</td>\n",
       "      <td>4.442167</td>\n",
       "      <td>3.150805</td>\n",
       "      <td>3.215227</td>\n",
       "      <td>2.830161</td>\n",
       "      <td>3.234261</td>\n",
       "      <td>3.544656</td>\n",
       "      <td>3.445095</td>\n",
       "      <td>2.869693</td>\n",
       "      <td>1.603221</td>\n",
       "      <td>2.699854</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>6.206440e+05</td>\n",
       "      <td>2.820761</td>\n",
       "      <td>3.065145</td>\n",
       "      <td>2.988581</td>\n",
       "      <td>2.864562</td>\n",
       "      <td>2.223085</td>\n",
       "      <td>3.643857</td>\n",
       "      <td>2.449697</td>\n",
       "      <td>3.052666</td>\n",
       "      <td>1.732674</td>\n",
       "      <td>0.954592</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>6.337500e+04</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>8.776170e+05</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.171795e+06</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.238705e+06</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.345435e+07</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Sample code number  Clump Thickness  Uniformity of Cell Size  \\\n",
       "count        6.830000e+02       683.000000               683.000000   \n",
       "mean         1.076720e+06         4.442167                 3.150805   \n",
       "std          6.206440e+05         2.820761                 3.065145   \n",
       "min          6.337500e+04         1.000000                 1.000000   \n",
       "25%          8.776170e+05         2.000000                 1.000000   \n",
       "50%          1.171795e+06         4.000000                 1.000000   \n",
       "75%          1.238705e+06         6.000000                 5.000000   \n",
       "max          1.345435e+07        10.000000                10.000000   \n",
       "\n",
       "       Uniformity of Cell Shape  Marginal Adhesion  \\\n",
       "count                683.000000         683.000000   \n",
       "mean                   3.215227           2.830161   \n",
       "std                    2.988581           2.864562   \n",
       "min                    1.000000           1.000000   \n",
       "25%                    1.000000           1.000000   \n",
       "50%                    1.000000           1.000000   \n",
       "75%                    5.000000           4.000000   \n",
       "max                   10.000000          10.000000   \n",
       "\n",
       "       Single Epithelial Cell Size  Bare Nuclei  Bland Chromatin  \\\n",
       "count                   683.000000   683.000000       683.000000   \n",
       "mean                      3.234261     3.544656         3.445095   \n",
       "std                       2.223085     3.643857         2.449697   \n",
       "min                       1.000000     1.000000         1.000000   \n",
       "25%                       2.000000     1.000000         2.000000   \n",
       "50%                       2.000000     1.000000         3.000000   \n",
       "75%                       4.000000     6.000000         5.000000   \n",
       "max                      10.000000    10.000000        10.000000   \n",
       "\n",
       "       Normal Nucleoli     Mitoses       Class  \n",
       "count       683.000000  683.000000  683.000000  \n",
       "mean          2.869693    1.603221    2.699854  \n",
       "std           3.052666    1.732674    0.954592  \n",
       "min           1.000000    1.000000    2.000000  \n",
       "25%           1.000000    1.000000    2.000000  \n",
       "50%           1.000000    1.000000    2.000000  \n",
       "75%           4.000000    1.000000    4.000000  \n",
       "max          10.000000   10.000000    4.000000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sample code number             False\n",
       "Clump Thickness                False\n",
       "Uniformity of Cell Size        False\n",
       "Uniformity of Cell Shape       False\n",
       "Marginal Adhesion              False\n",
       "Single Epithelial Cell Size    False\n",
       "Bare Nuclei                    False\n",
       "Bland Chromatin                False\n",
       "Normal Nucleoli                False\n",
       "Mitoses                        False\n",
       "Class                          False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (10,10))\n",
    "sns.heatmap(dataset.corr(), annot=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1000025,  1002945,  1015425,  1016277,  1017023,  1017122,\n",
       "        1018099,  1018561,  1033078,  1035283,  1036172,  1041801,\n",
       "        1043999,  1044572,  1047630,  1048672,  1049815,  1050670,\n",
       "        1050718,  1054590,  1054593,  1056784,  1059552,  1065726,\n",
       "        1066373,  1066979,  1067444,  1070935,  1071760,  1072179,\n",
       "        1074610,  1075123,  1079304,  1080185,  1081791,  1084584,\n",
       "        1091262,  1099510,  1100524,  1102573,  1103608,  1103722,\n",
       "        1105257,  1105524,  1106095,  1106829,  1108370,  1108449,\n",
       "        1110102,  1110503,  1110524,  1111249,  1112209,  1113038,\n",
       "        1113483,  1113906,  1115282,  1115293,  1116116,  1116132,\n",
       "        1116192,  1116998,  1117152,  1118039,  1120559,  1121732,\n",
       "        1121919,  1123061,  1124651,  1125035,  1126417,  1131294,\n",
       "        1132347,  1133041,  1133136,  1136142,  1137156,  1143978,\n",
       "        1147044,  1147699,  1147748,  1148278,  1148873,  1152331,\n",
       "        1155546,  1156272,  1156948,  1157734,  1158247,  1160476,\n",
       "        1164066,  1165297,  1165790,  1165926,  1166630,  1166654,\n",
       "        1167439,  1167471,  1168359,  1168736,  1169049,  1170419,\n",
       "        1170420,  1171710,  1171795,  1171845,  1172152,  1173216,\n",
       "        1173235,  1173347,  1173509,  1173514,  1173681,  1174057,\n",
       "        1174131,  1174428,  1175937,  1176406,  1176881,  1177027,\n",
       "        1177399,  1177512,  1178580,  1179818,  1180194,  1180523,\n",
       "        1180831,  1181356,  1182404,  1182410,  1183240,  1183516,\n",
       "        1183911,  1183983,  1184184,  1184241,  1185609,  1185610,\n",
       "        1187457,  1187805,  1188472,  1189266,  1189286,  1190394,\n",
       "        1190485,  1192325,  1193091,  1193210,  1196295,  1196915,\n",
       "        1197080,  1197270,  1197440,  1197979,  1197993,  1198128,\n",
       "        1198641,  1199219,  1199731,  1199983,  1200772,  1200847,\n",
       "        1200892,  1200952,  1201834,  1201936,  1202125,  1202812,\n",
       "        1203096,  1204242,  1204898,  1205138,  1205579,  1206089,\n",
       "        1206695,  1206841,  1207986,  1208301,  1210963,  1211202,\n",
       "        1212232,  1212251,  1212422,  1213375,  1213383,  1214092,\n",
       "        1214556,  1214966,  1216694,  1216947,  1217051,  1217264,\n",
       "        1218105,  1218741,  1218860,  1219406,  1219525,  1219859,\n",
       "        1220330,  1221863,  1222047,  1222936,  1223282,  1223426,\n",
       "        1223793,  1223967,  1224329,  1225799,  1226012,  1226612,\n",
       "        1227210,  1227244,  1227481,  1228152,  1228311,  1230175,\n",
       "        1230688,  1231387,  1231706,  1232225,  1236043,  1241559,\n",
       "        1241679,  1242364,  1243256,  1270479,  1276091,  1277018,\n",
       "         128059,  1285531,  1287775,   144888,   145447,   167528,\n",
       "         183913,   191250,   242970,   255644,   263538,   274137,\n",
       "         303213,   314428,   320675,   324427,   385103,   390840,\n",
       "         411453,   428903,   431495,   434518,   452264,   456282,\n",
       "         476903,   486283,   486662,   488173,   492268,   508234,\n",
       "         527363,   529329,   535331,   543558,   555977,   560680,\n",
       "         561477,   601265,   606722,   616240,   625201,    63375,\n",
       "         635844,   636130,   640744,   646904,   653777,   659642,\n",
       "         666090,   666942,   667204,   673637,   684955,   688033,\n",
       "         691628,   693702,   704097,   706426,   709287,   718641,\n",
       "         721482,   730881,   733639,   733823,   740492,   743348,\n",
       "         752904,   756136,   760001,   760239,    76389,   764974,\n",
       "         770066,   785208,   785615,   792744,   797327,   798429,\n",
       "         806423,   809912,   810104,   814265,   814911,   822829,\n",
       "         826923,   830690,   831268,   832226,   832567,   836433,\n",
       "         837082,   846832,   850831,   855524,   857774,   859164,\n",
       "         859350,   866325,   873549,   877291,   877943,   888169,\n",
       "         888523,   896404,   897172,    95719,   160296,   342245,\n",
       "         428598,   492561,   493452,   521441,   636437,   640712,\n",
       "         654244,   657753,   685977,   805448,   846423,  1002504,\n",
       "        1022257,  1026122,  1071084,  1080233,  1114570,  1116715,\n",
       "        1131411,  1151734,  1156017,  1158405,  1168278,  1176187,\n",
       "        1196263,  1196475,  1206314,  1211265,  1213784,  1223003,\n",
       "        1223306,  1223543,  1229929,  1231853,  1234554,  1236837,\n",
       "        1237674,  1238021,  1238633,  1238915,  1238948,  1239232,\n",
       "        1239347,  1239967,  1240337,  1253505,  1255384,  1257200,\n",
       "        1257648,  1257815,  1257938,  1258549,  1258556,  1266154,\n",
       "        1272039,  1277629,  1293439,  1294562,  1295186,   527337,\n",
       "         558538,   566509,   608157,   677910,   734111,   780555,\n",
       "         827627,  1049837,  1058849,  1193544,  1201870,  1202253,\n",
       "        1227081,  1230994,  1238410,  1246562,  1257470,  1259008,\n",
       "        1266124,  1267898,  1268313,  1268804,  1280258,  1293966,\n",
       "        1296572,  1298416,  1299596,  1181685,  1211594,  1238777,\n",
       "        1257608,  1269574,  1277145,  1287282,  1296025,  1296263,\n",
       "        1296593,  1299161,  1301945,  1302428,  1318169,   474162,\n",
       "         787451,  1002025,  1070522,  1073960,  1076352,  1084139,\n",
       "        1119189,  1133991,  1142706,  1155967,  1170945,  1181567,\n",
       "        1204558,  1217952,  1224565,  1238186,  1253917,  1265899,\n",
       "        1268766,  1277268,  1286943,  1295508,  1297327,  1297522,\n",
       "        1298360,  1299924,  1299994,  1304595,  1306282,  1313325,\n",
       "        1320077,  1320304,  1330439,   333093,   369565,   412300,\n",
       "         672113,   749653,   769612,   807657,  8233704,   837480,\n",
       "         867392,   869828,  1043068,  1056171,  1061990,  1113061,\n",
       "        1135090,  1145420,  1158157,  1171578,  1174841,  1184586,\n",
       "        1186936,  1197527,  1222464,  1240603,  1241035,  1287971,\n",
       "        1289391,  1306339,  1313658,  1313982,  1321264,  1321321,\n",
       "        1321348,  1321931,  1321942,  1328331,  1328755,  1331405,\n",
       "        1331412,  1333104,  1334071,  1343068,  1343374,  1344121,\n",
       "         142932,   183936,   324382,   378275,   690557,   695091,\n",
       "         695219,   824249,   871549,   878358,  1107684,  1115762,\n",
       "        1217717,  1239420,  1254538,  1261751,  1268275,  1272166,\n",
       "        1294261,  1295529,  1298484,  1311875,  1315506,  1320141,\n",
       "        1325309,  1333063,  1333495,  1334659,  1336798,  1344449,\n",
       "        1350568,  1352663,   188336,   352431,   353098,   557583,\n",
       "         636375,   736150,   803531,  1016634,  1031608,  1041043,\n",
       "        1042252,  1073836,  1083817,  1096352,  1140597,  1149548,\n",
       "        1174009,  1183596,  1190386,  1190546,  1213273,  1218982,\n",
       "        1225382,  1235807,  1253955,  1257366,  1260659,  1268952,\n",
       "        1275807,  1277792,  1285722,  1288608,  1290203,  1294413,\n",
       "        1303489,  1311033,  1311108,  1315807,  1318671,  1319609,\n",
       "        1323477,  1324572,  1324681,  1325159,  1326892,  1330361,\n",
       "        1333877,  1334015,  1334667,  1339781, 13454352,  1345452,\n",
       "        1345593,  1347749,  1347943,  1348851,  1350319,  1350423,\n",
       "        1352848,  1353092,  1354840,  1355260,  1365075,  1365328,\n",
       "        1368267,  1368273,  1368882,  1369821,  1371026,  1371920,\n",
       "         466906,   534555,   536708,   566346,   603148,   654546,\n",
       "         714039,   763235,   776715,   841769,   888820,   897471],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset[\"Sample code number\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 35 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dataset.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:, 1:-1].values\n",
    "y = dataset.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(683, 9) (683,)\n"
     ]
    }
   ],
   "source": [
    "print(X.shape,y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import the f_classif function and feed in the features and response:\n",
    "from sklearn.feature_selection import f_classif\n",
    "[f_stat, f_p_value] = f_classif(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature</th>\n",
       "      <th>F statistic</th>\n",
       "      <th>p value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Bare Nuclei</td>\n",
       "      <td>1426.240270</td>\n",
       "      <td>3.401103e-169</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uniformity of Cell Shape</td>\n",
       "      <td>1417.643841</td>\n",
       "      <td>1.369425e-168</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Uniformity of Cell Size</td>\n",
       "      <td>1406.132470</td>\n",
       "      <td>8.922226e-168</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Bland Chromatin</td>\n",
       "      <td>921.010015</td>\n",
       "      <td>1.267712e-128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Normal Nucleoli</td>\n",
       "      <td>727.470805</td>\n",
       "      <td>1.465645e-109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Clump Thickness</td>\n",
       "      <td>711.423446</td>\n",
       "      <td>7.292504e-108</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Marginal Adhesion</td>\n",
       "      <td>677.878400</td>\n",
       "      <td>2.979778e-104</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Single Epithelial Cell Size</td>\n",
       "      <td>622.157681</td>\n",
       "      <td>4.733540e-98</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Mitoses</td>\n",
       "      <td>148.787689</td>\n",
       "      <td>4.304040e-31</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       Feature  F statistic        p value\n",
       "5                  Bare Nuclei  1426.240270  3.401103e-169\n",
       "2     Uniformity of Cell Shape  1417.643841  1.369425e-168\n",
       "1      Uniformity of Cell Size  1406.132470  8.922226e-168\n",
       "6              Bland Chromatin   921.010015  1.267712e-128\n",
       "7              Normal Nucleoli   727.470805  1.465645e-109\n",
       "0              Clump Thickness   711.423446  7.292504e-108\n",
       "3            Marginal Adhesion   677.878400  2.979778e-104\n",
       "4  Single Epithelial Cell Size   622.157681   4.733540e-98\n",
       "8                      Mitoses   148.787689   4.304040e-31"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#create a DataFrame of feature names, F-statistics, and p-values, and show it sorted on p-value:\n",
    "f_test_df = pd.DataFrame({'Feature':dataset.columns[1:-1],\n",
    "                          'F statistic':f_stat,\n",
    "                          'p value':f_p_value})\n",
    "f_test_df.sort_values('p value')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SelectPercentile(percentile=50)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import the SelectPercentile class:\n",
    "#To select the top 20% of features according to the F-test\n",
    "from sklearn.feature_selection import SelectPercentile\n",
    "#Use the .fit method to fit the object on our features and response data, similar to how a model would be fit:\n",
    "selector = SelectPercentile(f_classif, percentile=50)\n",
    "selector.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False,  True,  True, False, False,  True,  True, False, False])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Capture the indices of the selected features in an array named best_feature_ix:\n",
    "best_feature_ix = selector.get_support()\n",
    "best_feature_ix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#The feature names can be obtained using all but the last element (the response variable name) of our features_response list by indexing with :-1:\n",
    "features = dataset.columns[1:-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Uniformity of Cell Size',\n",
       " 'Uniformity of Cell Shape',\n",
       " 'Bare Nuclei',\n",
       " 'Bland Chromatin']"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Find the selected feature names\n",
    "best_features = [features[counter] for counter in range(len(features))\n",
    "                 if best_feature_ix[counter]]\n",
    "best_features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "0AnzJQCj7TLO"
   },
   "source": [
    "## Splitting the dataset into the Training set and Test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "WOQwyng57dp2"
   },
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(546, 9)\n",
      "(137, 9)\n",
      "(546,)\n",
      "(137,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(Y_train.shape)\n",
    "print(Y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({4: 189, 2: 357})"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from collections import Counter\n",
    "Counter(Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "pS1Q-n_A7iZ_"
   },
   "source": [
    "## Create Baseline model on the Training set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "models_list = []\n",
    "models_list.append(('CART', DecisionTreeClassifier()))\n",
    "models_list.append(('SVM', SVC())) \n",
    "models_list.append(('LR', LogisticRegression()))\n",
    "models_list.append(('NB', GaussianNB()))\n",
    "models_list.append(('KNN', KNeighborsClassifier()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\hp\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\model_selection\\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.\n",
      "  FutureWarning\n",
      "c:\\users\\hp\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\model_selection\\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.\n",
      "  FutureWarning\n",
      "c:\\users\\hp\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\model_selection\\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.\n",
      "  FutureWarning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CART: 0.939596 (0.036589) (run time: 0.044009)\n",
      "SVM: 0.967071 (0.030354) (run time: 0.057279)\n",
      "LR: 0.967003 (0.023024) (run time: 0.233286)\n",
      "NB: 0.965219 (0.027777) (run time: 0.024002)\n",
      "KNN: 0.974377 (0.024814) (run time: 0.080693)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\hp\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\model_selection\\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.\n",
      "  FutureWarning\n",
      "c:\\users\\hp\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\model_selection\\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.\n",
      "  FutureWarning\n"
     ]
    }
   ],
   "source": [
    "num_folds = 10\n",
    "results = []\n",
    "names = []\n",
    "\n",
    "for name, model in models_list:\n",
    "    kfold = KFold(n_splits=num_folds, random_state=123)\n",
    "    start = time.time()\n",
    "    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')\n",
    "    end = time.time()\n",
    "    results.append(cv_results)\n",
    "    names.append(name)\n",
    "    print( \"%s: %f (%f) (run time: %f)\" % (name, cv_results.mean(), cv_results.std(), end-start))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "fig.suptitle('Performance Comparison')\n",
    "ax = fig.add_subplot(111)\n",
    "plt.boxplot(results)\n",
    "ax.set_xticklabels(names)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Evalution of Algorithms on Standardised Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "\n",
    "# Standardize the dataset\n",
    "pipelines = []\n",
    "\n",
    "pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeClassifier())])))\n",
    "pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))\n",
    "pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression())])))\n",
    "pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))\n",
    "pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ScaledCART: 0.932256 (0.023137) (run time: 0.068694)\n",
      "ScaledSVM: 0.967037 (0.029355) (run time: 0.074794)\n",
      "ScaledNB: 0.965152 (0.020947) (run time: 0.103971)\n",
      "ScaledNB: 0.965219 (0.027777) (run time: 0.042730)\n",
      "ScaledKNN: 0.967003 (0.028188) (run time: 0.109148)\n"
     ]
    }
   ],
   "source": [
    "results = []\n",
    "names = []\n",
    "with warnings.catch_warnings():\n",
    "    warnings.simplefilter(\"ignore\")\n",
    "    kfold = KFold(n_splits=num_folds, random_state=123)\n",
    "    for name, model in pipelines:\n",
    "        start = time.time()\n",
    "        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')\n",
    "        end = time.time()\n",
    "        results.append(cv_results)\n",
    "        names.append(name)\n",
    "        print( \"%s: %f (%f) (run time: %f)\" % (name, cv_results.mean(), cv_results.std(), end-start))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "fig.suptitle('Performance Comparison')\n",
    "ax = fig.add_subplot(111)\n",
    "plt.boxplot(results)\n",
    "ax.set_xticklabels(names)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hyperparameter Tuning - Tuning SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\hp\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\sklearn\\model_selection\\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.\n",
      "  FutureWarning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best: 0.974343 using {'C': 0.1, 'kernel': 'sigmoid'}\n",
      "0.968855 (0.024791) with: {'C': 0.1, 'kernel': 'linear'}\n",
      "0.926599 (0.043658) with: {'C': 0.1, 'kernel': 'poly'}\n",
      "0.968855 (0.024791) with: {'C': 0.1, 'kernel': 'rbf'}\n",
      "0.974343 (0.023539) with: {'C': 0.1, 'kernel': 'sigmoid'}\n",
      "0.968855 (0.024791) with: {'C': 0.3, 'kernel': 'linear'}\n",
      "0.948653 (0.028169) with: {'C': 0.3, 'kernel': 'poly'}\n",
      "0.967037 (0.027009) with: {'C': 0.3, 'kernel': 'rbf'}\n",
      "0.970673 (0.025008) with: {'C': 0.3, 'kernel': 'sigmoid'}\n",
      "0.967003 (0.023024) with: {'C': 0.5, 'kernel': 'linear'}\n",
      "0.954141 (0.026366) with: {'C': 0.5, 'kernel': 'poly'}\n",
      "0.968855 (0.028512) with: {'C': 0.5, 'kernel': 'rbf'}\n",
      "0.972525 (0.020601) with: {'C': 0.5, 'kernel': 'sigmoid'}\n",
      "0.967003 (0.023024) with: {'C': 0.7, 'kernel': 'linear'}\n",
      "0.954141 (0.026366) with: {'C': 0.7, 'kernel': 'poly'}\n",
      "0.968855 (0.028512) with: {'C': 0.7, 'kernel': 'rbf'}\n",
      "0.966970 (0.024549) with: {'C': 0.7, 'kernel': 'sigmoid'}\n",
      "0.967003 (0.023024) with: {'C': 0.9, 'kernel': 'linear'}\n",
      "0.954141 (0.026366) with: {'C': 0.9, 'kernel': 'poly'}\n",
      "0.967037 (0.029355) with: {'C': 0.9, 'kernel': 'rbf'}\n",
      "0.967003 (0.023024) with: {'C': 0.9, 'kernel': 'sigmoid'}\n",
      "0.967003 (0.023024) with: {'C': 1.0, 'kernel': 'linear'}\n",
      "0.955960 (0.028733) with: {'C': 1.0, 'kernel': 'poly'}\n",
      "0.967037 (0.029355) with: {'C': 1.0, 'kernel': 'rbf'}\n",
      "0.965185 (0.023919) with: {'C': 1.0, 'kernel': 'sigmoid'}\n",
      "0.965152 (0.020947) with: {'C': 1.3, 'kernel': 'linear'}\n",
      "0.957811 (0.032837) with: {'C': 1.3, 'kernel': 'poly'}\n",
      "0.965219 (0.032188) with: {'C': 1.3, 'kernel': 'rbf'}\n",
      "0.963333 (0.027357) with: {'C': 1.3, 'kernel': 'sigmoid'}\n",
      "0.965152 (0.020947) with: {'C': 1.5, 'kernel': 'linear'}\n",
      "0.959630 (0.029406) with: {'C': 1.5, 'kernel': 'poly'}\n",
      "0.963367 (0.033821) with: {'C': 1.5, 'kernel': 'rbf'}\n",
      "0.963333 (0.027357) with: {'C': 1.5, 'kernel': 'sigmoid'}\n",
      "0.965152 (0.020947) with: {'C': 1.7, 'kernel': 'linear'}\n",
      "0.959630 (0.029406) with: {'C': 1.7, 'kernel': 'poly'}\n",
      "0.963367 (0.033821) with: {'C': 1.7, 'kernel': 'rbf'}\n",
      "0.959697 (0.028268) with: {'C': 1.7, 'kernel': 'sigmoid'}\n",
      "0.965152 (0.020947) with: {'C': 2.0, 'kernel': 'linear'}\n",
      "0.961481 (0.026498) with: {'C': 2.0, 'kernel': 'poly'}\n",
      "0.961549 (0.032255) with: {'C': 2.0, 'kernel': 'rbf'}\n",
      "0.956027 (0.028707) with: {'C': 2.0, 'kernel': 'sigmoid'}\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "scaler = StandardScaler().fit(X_train)\n",
    "rescaledX = scaler.transform(X_train)\n",
    "c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]\n",
    "kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']\n",
    "param_grid = dict(C=c_values, kernel=kernel_values)\n",
    "model = SVC()\n",
    "kfold = KFold(n_splits=num_folds, random_state=21)\n",
    "grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)\n",
    "grid_result = grid.fit(rescaledX, Y_train)\n",
    "print(\"Best: %f using %s\" % (grid_result.best_score_, grid_result.best_params_))\n",
    "means = grid_result.cv_results_['mean_test_score']\n",
    "stds = grid_result.cv_results_['std_test_score']\n",
    "params = grid_result.cv_results_['params']\n",
    "for mean, stdev, param in zip(means, stds, params):\n",
    "    print(\"%f (%f) with: %r\" % (mean, stdev, param))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Application of SVC on dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Run Time: 0.004140\n"
     ]
    }
   ],
   "source": [
    "# prepare the model\n",
    "with warnings.catch_warnings():\n",
    "    warnings.simplefilter(\"ignore\")\n",
    "    scaler = StandardScaler().fit(X_train)\n",
    "X_train_scaled = scaler.transform(X_train)\n",
    "model = SVC(C = 0.1, kernel = 'sigmoid')\n",
    "start = time.time()\n",
    "model.fit(X_train_scaled,Y_train)\n",
    "end = time.time()\n",
    "print( \"Run Time: %f\" % (end-start))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "4cgD7EnB8Dnd"
   },
   "source": [
    "## Predicting the Test set results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "1B4zQvOq8M7H"
   },
   "outputs": [],
   "source": [
    "# estimate accuracy on test dataset\n",
    "with warnings.catch_warnings():\n",
    "    warnings.simplefilter(\"ignore\")\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "predictions = model.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.948905109489051"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score = model.score(X_test_scaled,Y_test)\n",
    "score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "26CHkZbs8Tu5"
   },
   "source": [
    "## Making the Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 50
    },
    "colab_type": "code",
    "id": "waJZi8fw8m_2",
    "outputId": "985787e6-54db-42ca-edde-950ade8d1ac6"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 94.89 %\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           2       0.97      0.95      0.96        87\n",
      "           4       0.92      0.94      0.93        50\n",
      "\n",
      "    accuracy                           0.95       137\n",
      "   macro avg       0.94      0.95      0.95       137\n",
      "weighted avg       0.95      0.95      0.95       137\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy: {:.2f} %\".format(score*100))\n",
    "print(classification_report(Y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "cm = confusion_matrix(Y_test, predictions)\n",
    "plt.figure(figsize=(4,4))\n",
    "sns.heatmap(cm, annot=True, fmt=\".3f\", linewidths=.5, square = True, cmap = 'Blues_r');\n",
    "plt.ylabel('Actual label');\n",
    "plt.xlabel('Predicted label');\n",
    "all_sample_title = 'Accuracy Score: {:.2f}%'.format(model.score(X_test_scaled,Y_test)*100)\n",
    "plt.title(all_sample_title, size = 15);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Logistic Regression",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
