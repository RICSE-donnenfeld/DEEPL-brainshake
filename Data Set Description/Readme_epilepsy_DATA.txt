MIT epilepsy DATA

Introduction:

The MIT epilepsy RAW data originally has:
* 24 patients
* For each patient, a directory with a series of files with consecutive recordings, 23 channels and the initial and final time. For data protection reasons, date is not known
* The channels are not directly the sensors but the difference between two consecutive sensor measurements combined as specified in figure 3 from DataDescription.pdf

The data has been processed such that:
1. All patient data is contained in a single file in the raw directory
2. An excel file has been generated with the information of this data that is in the annot directory: df_annotation_full.xlsx

This file has the following columns for each .edf file:
1. type: indicates whether the .edf file contains an attack: normal, if there is no attack, seizure, if there is any
2. PatID: patient identifier
3. filename: .edf filename that identifies the recording
4.seizure_id: number that identifies the attack in the edf file, 0 if there is no attack, and the attack order in case there is more than one (e.g. if there are several consecutive attacks in the file it would be a 1 for the first a 2 for the second, etc.).
5. seizure_start: time (seconds) the attack begins in the .edf recording
6. seizure_end: : time (seconds) the attack ends in the .edf recording
7. rec_duration: duration of the .edf recording (seconds)
8. seizure_duration: duration (seconds) of each attack