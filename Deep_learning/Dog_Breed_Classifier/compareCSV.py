
import difflib
import csv as csv

INPUT_PATH = "C:/Users/Divya moorjaney/Documents/FIS ADA/CompareCSV/"

def comparecsv(sqlfile, scalafile, split):
    d = difflib.Differ()

    sql_open = open(INPUT_PATH + sqlfile, 'r')  # open file
    sql_colcount = len(sql_open.readline().strip().split(split))  # get col counts ## TODO This will only check for number of cols in the first line, what if there is a delimiter mismatch in some other line in the file?
    sql = csv.reader(sql_open)  # read file
    sql_rowcount = len(list(sql))  # count of rows in the file
    # sql_open.close()

    scala_open = open(INPUT_PATH + scalafile, 'r')
    scala_colcount = len(scala_open.readline().strip().split(split)) ## TODO This will only check for number of cols in the first line, what if there is a delimiter mismatch in some other line in the file?
    scala = csv.reader(scala_open)
    scala_rowcount = len(list(scala))
    # scala_open.close()

    if sql_rowcount != scala_rowcount or sql_colcount != scala_colcount:
        print('Row and column counts do not match')
        print('Number of columns in SQL output is '+str(sql_colcount))
        print('Number of columns in Scala output is ' +str(scala_colcount))
        print('Number of rows in SQL output is ' +str(sql_rowcount))
        print('Number of rows in Scala output is ' +str(scala_rowcount))

        ##  TODO the program should exit safely at this stage and do nothing else

    else:
       # print('Row counts and column counts match')
       # outFile = open('difference_sql_scala.csv', 'w')
        for line in sql:    ## TODO: Why are we doing this? Will be extremely expensive for larger files
            #  result = d.compare(line1, line2)
            print('|'.join(line))

        ##  TODO: What happens after this? The later portions are all commented out

# sqlTraining = pd.read_csv(r"C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\contact_attrition_108_20180321.csv", sep = ",")
# scalaTraining = pd.read_csv(r"C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\Contact_Attrition_Training.csv", sep = ",")

# print(scalaTraining.shape[0])   # to give the row count 181775
# print(scalaTraining.shape[1])   # to give column count  57
# print(scalaTraining.shape)      # gives the shape of the data frame (181775, 57)
# print(sqlTraining.shape)        # (151704, 57)


    # sqlRead = csv.reader(r"C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\contact_attrition_108_20180321.csv")
    # scalaRead = csv.reader(r"C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\Contact_Attrition_Training.csv")

    # sql = open(r'C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\contact_attrition_108_20180321.csv', 'r')
    # scala = open(r'C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\Contact_Attrition_Training.csv', 'r')
    # iterate each file by line - read in the file line by line
    # sqlRead = sql.readlines()   # change to read line by line
    # scalaRead = scala.readlines()
    # sql.close()
    # scala.close()

    # outFile = open('difference_sql_scala.csv', 'w')
    # x = 0
    # for i in sqlRead:
     #    if i != scalaRead[x]:
     #        outFile.write(scalaRead[x])
     #    x += 1
    # outFile.close()




# reader1 = csv.reader(r"C:\Users\Divya moorjaney\PycharmProjects\CompareCSV\venv\contact_attrition_108_20180321.csv")
# reader2

if __name__ == "__main__":
    comparecsv(sqlfile='sql.csv', scalafile='sql.csv', split='|')