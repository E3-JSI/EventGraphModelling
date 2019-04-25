from eventregistry import *
import json 

# Define path & name of the file for results
path = "../" # change
nameOfFile = "all_business_events_2018"

# Intialise ER 
er = EventRegistry(apiKey = "") # insert your own api key

# Dates
dateStart = "2018-01-01"
dateEnd = "2019-01-01"

# Search query 
q = QueryEventsIter(categoryUri = er.getCategoryUri("Business"),
                    dateStart =dateStart,
                    dateEnd =dateEnd
                    )

# Run query and save results
with open(path+nameOfFile + ".json", 'a') as outfile:
    for event in q.execQuery(er, sortBy = "date"):
        json.dump(event, outfile)
        outfile.write("\n")
