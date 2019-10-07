import eventregistry as ER
import pandas as pd 
import json 
import os
import pdb
import ast 

def get_er_data(er_user,comp,out_name,save_path,dateStart,dateEnd):
    """
    Get data from EventRegistry API

    Also check if part of data already downloaded and just append to that file

    :param er_user: EventRegistry class with your API key inside
    :param str comp: Concept for which we wish to get data
    :param str out_name: Name of the file to which data will be saved
    :param str save_path: Path to location where file will be saved
    :param str dateStart: Start date for downloading data, format "2019-01-30"
    :param str dateEnd: End date, same format as dateStart

    return: None, save file to save_path+out_name+".json"
    """
    # Define outfile
    outfile = save_path+out_name + ".json"
    if os.path.exists(outfile):
        with open(outfile) as outfile2:
            # Iterate through entire file
            for line in outfile2:
                pass
            # Read only the last line
            line1=json.loads(line)
            if pd.to_datetime(dateEnd) <= pd.to_datetime(line1["eventDate"]):
                print("Already have data")
                return 
            else:
                dateStart = line1["eventDate"]
                print("Starting at %s" % dateStart)

    comp_uri = er_user.getConceptUri(comp)
    # Check if you get uri, if not return nothing 
    if comp_uri is None: 
        print("No concept uri found")
        return 

    # Main query definition 
    query = ER.QueryEventsIter(conceptUri= comp_uri,
                    dateStart =dateStart,
                    dateEnd =dateEnd
                    )
    # Saving the file
    with open(outfile, 'a') as outfile:
        for event in query.execQuery(er_user, sortBy = "date",sortByAsc = True):
            try:
                print(event["eventDate"])
                outfile.write("\n")
                json.dump(event, outfile)
            except:
                print("Event has no date, moving on")
                continue

save_path = "../../Data/EventRegistry/"
stocks = pd.read_csv("../../Data/others/SP500constituents.csv")

api = "" # fill in your own

er_user = ER.EventRegistry(apiKey = api)
dateStart = "2014-01-01"
dateEnd = "2019-01-01"

if True:
    for tick,comp,_ in stocks.values:
        print(comp)
        get_er_data(er_user,comp,tick,save_path,dateStart,dateEnd)

if False:
    # Single concept download example
    print("Abbott")
    get_er_data(er_user,"ABT","ABT",save_path,dateStart,dateEnd)