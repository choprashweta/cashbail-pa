# cashbail-pa

## Introduction


## Dataset

The original dataset was purchased by the ACLU of Pennsylvania from the Administrative Office of Pennsylvania Courts (AOPC). It contains a complete record of bail assignment and offenses for every defendant charged with a crime within the state of Pennsylvania between January 1, 2016 and December 31, 2017. 

The dataset consisted of individual rows for each change included as part of an offence.

## Data Preparation

Before being used for analysis, the original AOPC dataset was cleaned and reduced to a single row per offence per defendent dataset. 

The data was subsetted to obtain a single record for each unique offense that was charged within the Magisterial District Court system at any point during 2016 and 2017. Because many defendants are charged with multiple crimes or charges under the same offense, this was essential to get a proper understanding of the actual amount and severity of crime in the state. This was possible through the use of the Offense Tracking Number (OTN) that is assigned by the court system to a defendant when they are first charged. All charges that fall under the same offense are labeled with the same OTN, allowing for the deduplication of records accordingly.

