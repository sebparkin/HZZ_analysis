
import infofile # local file containing cross-sections, sums of weights, dataset IDs
import numpy as np # for numerical calculations such as histogramming
import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import time
from HZZ_task import cut_data, calculate_mass, calculate_weight
from celery.result import AsyncResult
import os
from celery import group
import json

chunk_sizes = json.load(open('/app/data/chunk_sizes.txt'))

path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" 

variables = ['lep_pt','lep_eta','lep_phi','lep_E','lep_charge','lep_type']

weight_variables = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]

MeV = 0.001
GeV = 1.0

samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'], # data is from 2016, first four periods of data taking (ABCD)
    },

}

# Set luminosity to 10 fb-1 for all data
lumi = 10

# Controls the fraction of all events analysed
fraction = 1.0 # reduce this is if you want quicker runtime (implemented in the loop over the tree)

# Define empty dictionary to hold awkward arrays
all_data = {} 
total_elapsed = 0

#define number of chunks to split into

print('Starting...')
# Loop over samples
for s in samples: 
    # Print which sample is being processed
    print('Processing '+s+' samples') 

    # Define empty list to hold data
    frames = [] 

    # Loop over each file
    for val in samples[s]['list']: 
        if s == 'data': 
            prefix = "Data/" # Data prefix
        else: # MC prefix
            prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
        fileString = path+prefix+val+".4lep.root" # file name to open


        # start the clock
        start = time.time() 
        print("\t"+val+":") 

        # Open file
        tree = uproot.open(fileString + ":mini")
        
        sample_data = []

        # Loop over data in the tree
        for data in tree.iterate(variables + weight_variables, 
                                 library="ak", 
                                 entry_stop=tree.num_entries*fraction, # process up to numevents*fraction
                                 step_size = 1000000): 
            # Number of events in this batch
            nIn = len(data) 

            #Calculating number of chunks and creating a split index
            chunks = nIn//chunk_sizes['data']
            index = np.arange(len(data))
            split_index = np.array_split(index, chunks if chunks >= 1 else 1)
            
            #Cuts
            lep_type = data['lep_type']
            lep_charge = data['lep_charge']
            data2 = group(cut_data.s(data[split_index[i]], lep_type[split_index[i]], lep_charge[split_index[i]]) for i in range(chunks))
            result = data2.apply_async()
            data = result.join()
            data = ak.concatenate(data)
            
            #Redefining chunks for new data length
            chunks = len(data)//chunk_sizes['data']
            index = np.arange(len(data))
            split_index = np.array_split(index, chunks if chunks >= 1 else 1)

            #Invariant Mass
            mass2 = group(calculate_mass.s(data[split_index[i]]) for i in range(chunks))
            result = mass2.apply_async()
            mass = result.join()
            mass = np.concatenate(mass)
            data['mass'] = mass

            # Store Monte Carlo weights in the data
            if 'data' not in val: # Only calculates weights if the data is MC
                weight2 = group(calculate_weight.s(data[split_index[i]], val) for i in range(chunks))
                result = weight2.apply_async()
                weight = result.join()
                data['totalWeight'] = np.concatenate(weight)
                nOut = sum(data['totalWeight']) # sum of weights passing cuts in this batch 
            else:
                nOut = len(data)
            elapsed = time.time() - start # time taken to process
            total_elapsed += elapsed
            print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after

            # Append data to the whole sample data list
            sample_data.append(data)


        frames.append(ak.concatenate(sample_data)) 

    all_data = ak.concatenate(frames) # dictionary entry is concatenated awkward arrays


print(f'Total time: {round(total_elapsed, 2)}s')
os.makedirs('/app/data/', exist_ok=True)
filepath = os.path.join('/app/data/', 'data.parquet')
ak.to_parquet(all_data, filepath)

