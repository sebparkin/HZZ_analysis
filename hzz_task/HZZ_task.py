import infofile
import uproot
import time
from celery import Celery
import vector
import awkward as ak
import numpy as np

app = Celery('HZZ_task', broker = 'pyamqp://guest@rabbitmq//', backend = 'rpc://')

app.conf.update(
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle', 'json']  # Ensure workers accept both
)

# Set luminosity to 10 fb-1 for all data
lumi = 10

MeV = 0.001

# Cut lepton type (electron type is 11,  muon type is 13)
def cut_lep_type(lep_type):
    sum_lep_type = lep_type[:, 0] + lep_type[:, 1] + lep_type[:, 2] + lep_type[:, 3]
    lep_type_cut_bool = (sum_lep_type != 44) & (sum_lep_type != 48) & (sum_lep_type != 52)
    return lep_type_cut_bool # True means we should remove this entry (lepton type does not match)

# Cut lepton charge
def cut_lep_charge(lep_charge):
    # first lepton in each event is [:, 0], 2nd lepton is [:, 1] etc
    sum_lep_charge = lep_charge[:, 0] + lep_charge[:, 1] + lep_charge[:, 2] + lep_charge[:, 3] != 0
    return sum_lep_charge # True means we should remove this entry (sum of lepton charges is not equal to 0)

# Calculate invariant mass of the 4-lepton state
# [:, i] selects the i-th lepton in each event
def calc_mass(lep_pt, lep_eta, lep_phi, lep_E):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_E})
    invariant_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV # .M calculates the invariant mass
    return invariant_mass

def calc_weight(weight_variables, sample, events):
    info = infofile.infos[sample]
    xsec_weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    total_weight = xsec_weight 
    for variable in weight_variables:
        total_weight = total_weight * events[variable]
    return total_weight


@app.task
def cut_data(data, lep_type, lep_charge):
    data2 = data[~((cut_lep_type(lep_type) | cut_lep_charge(lep_charge)))]
    return(data2)

@app.task
def calculate_mass(data):
    mass = calc_mass(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_E'])
    return ak.to_list(mass)

@app.task
def calculate_weight(data, value):
    weight_variables = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]  
    split_data = {}
    for variable in weight_variables:
        split_data[variable] = data[variable]
    weight = calc_weight(weight_variables, value, split_data)
    return weight
    