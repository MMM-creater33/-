from geopy import distance
import searoute as sr
import numpy as np
import pandas as pd
import pickle
import os 
from countryinfo import CountryInfo
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

storage_days_by_transtype = {'Tanker': 3, 'Truck': 3, 'Pipeline': 0.5}

"""Define Classes"""


class results():
    # hold the final results of the model, inlcuding use impacts
    instances = {}

    def __init__(self, mcnum, pathwaynum, pathwaydefinition, pathwaytype, use, productionmeth, vectors, tinfra,
                 energysource, energytotal, emissionstotal, energy, emissions, _yield, emissionsfactor):
        self.mcnum = mcnum
        self.pathwaynum = pathwaynum  # pathway number
        self.pathwaydefinition = pathwaydefinition  # pathway stages
        self.pathwaytype = pathwaytype  # type (simple/domestic or energy to prod)
        self.energysource = energysource  # primary energy source
        self.use = use
        self.productionmeth = productionmeth
        self.vectors = vectors  # vector at each stage of the pathway
        self.tinfra = tinfra  # transmission infrastructure at each stage of the pathway
        self.storage_method = None
        self.emissions_total = emissionstotal
        self.energy_total = energytotal
        self.energy_by_stage = energy
        self.emissionsfactor = emissionsfactor
        self.emissions_by_stage = emissions
        self.use_impacts = final_use_impacts(self, uses.instances)
        self._yield = _yield
        self.instances[len(self.instances)] = self


class energy():
    # class to hold energy values for each stage of the pathway
    def __init__(self, embodied, process, operational):
        self.embodied = embodied
        self.process = process
        self.operational = operational
        self.total = None
        self.check_not_negative()

    def check_not_negative(self):
        for stage in [self.embodied, self.process, self.operational]:
            for energytype in stage:
                if stage[energytype] < 0:
                    print('Negative Energy Type', stage[energytype], energytype)
                    exit()

    def calc_energy(self, class_object, utilisation, esource, factor=1, tinfracapacity=None):
        # per unit energy use for the stage. Unit is defined in the main class but likely to be kWh energy transferred (LHV).
        # each of embodied, om and pr energies are strings that need to be converted into dictionaries
        # have not accounted for efficiency of energy production
        # esource is dependent on the stage of the pathway and pathway type; if it is after hydrogen production then assume
        lifetime = class_object.variables.lifetime
        if tinfracapacity == None:
            capacity = class_object.capacity
        else:
            capacity = tinfracapacity

        if capacity == 0 or lifetime == 0:
            print('capacity or lifetime is zero, check data')
            exit()

        if utilisation == 0:
            utilisation = 1
        # new dictionary to store the total energy use for each stage
        total_energy = {'Embodied': {}, 'O_M': {}, 'Process': {}, 'Total': {}}
        for energytype in self.embodied:
            total_energy['Embodied'][energytype] = self.embodied[energytype] * factor / (
                        lifetime * capacity * utilisation)

        for energytype in self.operational:
            total_energy['O_M'][energytype] = self.operational[energytype] * factor / (capacity * utilisation)

        for energytype in self.process:
            total_energy['Process'][energytype] = self.process[energytype] * factor

        total_energy['Total'] = 0
        for stage in total_energy:
            total = 0
            if stage != "Total":
                for energytype in total_energy[stage]:
                    if np.isnan(total_energy[stage][energytype]):
                        total_energy[stage][energytype] = 0
                    total += total_energy[stage][energytype]
                total_energy[stage]['Total'] = total
                total_energy['Total'] = total + total_energy['Total']

        return total_energy


class emissions():

    def __init__(self, embodied, process, operational, h2emissions):
        self.embodied = embodied
        self.process = process
        self.operational = operational
        self.h2emissions = h2emissions
        self.total = None

    def calc_emissions(self, emissionsfactors, energy, class_object, utilisation, factor, h2=True, tinfracapacity=None):
        lifetime = class_object.variables.lifetime
        if tinfracapacity == None or tinfracapacity < 0.00001:
            capacity = class_object.capacity
        else:
            capacity = tinfracapacity

        # new dictionary to store total emissions
        total_emissions = {}
        total_emissions['Embodied'] = self.embodied * factor / (lifetime * capacity * utilisation)
        total_emissions['O_M'] = self.operational * factor / (capacity * utilisation)
        if self.process < 0:
            print(self.process)
        total_emissions['Process'] = self.process * factor

        # add contributions from energy use
        for stage in ('Embodied', 'O_M', 'Process'):
            emissions = total_emissions[stage]
            for energytype in energy[stage]:
                if energytype in emissionsfactors.values():
                    e_factor = energytype.emissionsfactor
                    emissions_process = e_factor * energy[stage][energytype]
                    if emissions_process > 100:
                        print('Process Emissions more than 100', stage, energytype.ID, emissions_process,
                              class_object.ID, e_factor)
                        exit()
                elif energytype in emissionsfactors.keys():
                    e_factor = emissionsfactors[energytype].emissionsfactor
                    emissions_process = e_factor * energy[stage][energytype]
                else:
                    emissions_process = 0
                emissions += emissions_process
            total_emissions[stage] = emissions

        if total_emissions['Embodied'] > 100:
            print('embodied emissions more than 100', total_emissions['Embodied'], factor, lifetime, 'capacity',
                  capacity, utilisation, self.embodied)
            exit()

        if h2 == True:
            if np.isnan(self.h2emissions):
                self.h2emissions = 0
            total_emissions['H2 emissions'] = self.h2emissions * factor * 12
            if total_emissions['H2 emissions'] > 100:
                print('H2 Emissions more than 100', factor, self.h2emissions)
                exit()
        else:
            total_emissions['H2 emissions'] = 0

        total = 0
        for stage in total_emissions:
            total += total_emissions[stage]
        total_emissions['Total'] = total

        return total_emissions


class locations():
    instances = {}

    def __init__(self, ID, locationtype, offshoreonshore, country, lat, long, prodmethods, vectors, energytypes,
                 energysources, uses, maxsupply):
        self.ID = ID
        self.locationtype = locationtype
        self.offshoreonshore = offshoreonshore
        self.country = country
        self.lat = lat
        self.long = long
        self.prodmethods = prodmethods
        self.vectors = vectors
        self.energytype = energytypes
        self.energysources = energysources
        self.uses = uses
        self.maxsupply = maxsupply
        self.instances[self.ID] = self
        self.removespaces()

    def removespaces(self):
        for x in self.vectors:
            x = x.replace(" ", "")

    def energytypes(self, es_instances):
        # assign energy type to location
        for source in es_instances.values():
            if source.ID in self.energysources:
                self.energytype.append(source.energytype)
        return self.energytype


class H2_supply_chain():
    instances = {}

    def __init__(self, pathwaynum, pathwaydefinition, pathwaytype, productionmeth, vectors, tinfra, storage,
                 energysource=None):
        self.pathwaynum = pathwaynum
        self.pathwaydefinition = pathwaydefinition  # list of pathway stages (locations)
        self.pathwaytype = pathwaytype  # simple, energy to port, energy to prod
        self.energysource = energysource  # primary energy source
        self.productionmeth = productionmeth  # production method
        self.productioninfra = None
        self.vectors = vectors  # vectors, in order of pathway stages
        self.tinfra = tinfra  # transmission infra, in order of pathway sta
        self.end_uses = self.pathwaydefinition[-1].destination.uses
        self.storage_method = storage
        self.emissions_total = None
        self.energy_total = None
        self.energy = None
        self.emissions = None
        self.capacities = []
        self.instances[len(self.instances)] = self
        self.add_conversions()
        self.production_utilisation()
        self._yield = None
        # self.calc_yield()

        for i in range(len(self.tinfra)):
            if self.tinfra[i].trans_type in ("Tanker", "Truck", 'Pipeline') and self.pathwaydefinition[i].distance > 0:
                self.capacities.append(self.tinfra[i].capacity * 365 * 24 / (
                            self.pathwaydefinition[i].distance / self.tinfra[i].variables.speed))
            elif self.tinfra[i].trans_type in ("Tanker", "Truck", 'Pipeline'):
                # if set distance to 0.0001km
                self.capacities.append(self.tinfra[i].capacity * (365 * 24 / (0.0001 / self.tinfra[i].variables.speed)))
            else:
                self.capacities.append(self.tinfra[i].capacity)

    def production_utilisation(self):
        # calculate the utilisation of the production method based on the energy source

        # find initial capacity before base capacity changed to that of the energy source
        capacity = self.productionmeth.capacity / self.productionmeth.variables.capacity_factor

        # if utilisation factor is not zero (or non zero value in sobol), then change the capacity factor of the production method
        if self.energysource.utilisationfactor > 0.00001:
            self.productionmeth.variables.capacity_factor = self.energysource.utilisationfactor
            self.productionmeth.capacity = capacity * self.productionmeth.variables.capacity_factor

    def assign_energysources(self):

        # common to all pathways
        t1_fixed = assign_energysources_class(self.tinfra[0].variables.energy_fixed, self.energysource,
                                              self.pathwaydefinition[0].origin.country)
        t1_dyn = assign_energysources_class(self.tinfra[0].variables.energy_km, self.energysource,
                                            self.pathwaydefinition[0].origin.country)
        t2_fixed = assign_energysources_class(self.tinfra[1].variables.energy_fixed, self.energysource,
                                              self.pathwaydefinition[1].origin.country)
        t2_dyn = assign_energysources_class(self.tinfra[1].variables.energy_km, self.energysource,
                                            self.pathwaydefinition[1].origin.country)
        s1_fixed = assign_energysources_class(self.storage_method[0].variables.energy_fixed, self.energysource,
                                              self.pathwaydefinition[1].origin.country)
        prod_fixed = assign_energysources_class(self.productionmeth.variables.energy_fixed, self.energysource,
                                                self.pathwaydefinition[0].origin.country)
        prod_dyn = assign_energysources_class(self.productionmeth.variables.energy_km, self.energysource,
                                              self.pathwaydefinition[0].origin.country)
        if self.conversions[0] in conversions.instances.values():
            conv = assign_energysources_class(self.conversions[0].variables.energy_fixed, self.energysource,
                                              self.pathwaydefinition[0].origin.country)
        else:
            if self.conversions[0].ID in conversions.instances.keys():
                conv = assign_energysources_class(self.conversions[0].variables.energy_fixed, self.energysource,
                                                  self.pathwaydefinition[0].origin.country)
            else:
                conv = 0

        # energy to prod
        if self.pathwaytype != "Simple":
            t3_fixed = assign_energysources_class(self.tinfra[2].variables.energy_fixed, self.energysource,
                                                  self.pathwaydefinition[2].origin.country)
            t3_dyn = assign_energysources_class(self.tinfra[2].variables.energy_km, self.energysource,
                                                self.pathwaydefinition[2].origin.country)
            t4_fixed = assign_energysources_class(self.tinfra[3].variables.energy_fixed, self.energysource,
                                                  self.pathwaydefinition[3].origin.country)
            t4_dyn = assign_energysources_class(self.tinfra[3].variables.energy_km, self.energysource,
                                                self.pathwaydefinition[3].origin.country)
            s2_fixed = assign_energysources_class(self.storage_method[1].variables.energy_fixed, self.energysource,
                                                  self.pathwaydefinition[2].origin.country)
            if self.conversions[1] in conversions.instances.values():
                conv2 = assign_energysources_class(self.conversions[1].variables.energy_fixed, self.energysource,
                                                   self.pathwaydefinition[2].origin.country)
            else:
                conv2 = 0
            if self.conversions[2] in conversions.instances.values():
                conv3 = assign_energysources_class(self.conversions[2].variables.energy_fixed, self.energysource,
                                                   self.pathwaydefinition[3].origin.country)
            else:
                conv3 = 0
        # simple
        else:
            t3_fixed = 0
            t4_fixed = 0
            t3_dyn = 0
            t4_dyn = 0
            conv2 = 0
            conv3 = 0
            s2_fixed = 0

        assignedenergies = {'Prod Fixed': prod_fixed, 'Prod Dyn': prod_dyn, 'T1 Fixed': t1_fixed, 'T1 Dyn': t1_dyn,
                            'T2 Fixed': t2_fixed, 'T2 Dyn': t2_dyn, 'T3 Fixed': t3_fixed, 'T4 Fixed': t4_fixed,
                            'T3 Dyn': t3_dyn,
                            'T4 Dyn': t4_dyn, 'Conv': conv, 'Conv2': conv2, 'Conv3': conv3, 'Stor1': s1_fixed,
                            'Stor2': s2_fixed}

        return assignedenergies

    def add_conversions(self):
        conversionslist = []
        for num in range(len(self.vectors) - 1):
            # double check no spaces remain in string
            vector1 = self.vectors[num].replace(" ", "")
            vector2 = self.vectors[num + 1].replace(" ", "")
            if vector1 in ("NG", "E") and vector2 not in ("NG", "E"):
                conversionslist.append(conversions.instances['CH2-' + vector2])
            else:
                conversionslist.append(conversions.instances[vector1 + '-' + vector2])
        self.conversions = conversionslist

    def calc_yield(self):
        # calculate the yield for all the conversion stages
        conversionyields = []
        for conversion in self.conversions:
            conversionyields.append(conversion.variables._yield)

        stage1 = self.tinfra[0].variables._yield * (
                    1 - self.tinfra[0].variables._loss_km * self.pathwaydefinition[0].distance)
        stage2 = self.productionmeth.variables._yield  # accounted for in energy demand calculation, so unused
        stage3 = conversionyields[0]

        if self.pathwaytype == "Energy to Prod":
            stage4 = self.tinfra[1].variables._yield * (
                        1 - self.tinfra[1].variables._loss_km * self.pathwaydefinition[1].distance)
            stage5 = conversionyields[1]
            transtype = self.tinfra[2].trans_type
            stage6 = self.storage_method[0].variables._yield * (
                        1 - self.storage_method[0].variables._loss_day * storage_days_by_transtype[transtype])
            stage7 = self.tinfra[2].variables._yield * (
                        1 - self.tinfra[2].variables._loss_km * self.pathwaydefinition[2].distance)
            stage8 = conversionyields[2]
            transtype = self.tinfra[3].trans_type
            stage9 = self.storage_method[1].variables._yield * (
                        1 - self.storage_method[1].variables._loss_day * storage_days_by_transtype[transtype])
            stage10 = self.tinfra[3].variables._yield * (
                        1 - self.tinfra[3].variables._loss_km * self.pathwaydefinition[3].distance)
            self._yield = {1: 1 / stage1,
                           # primary energy transmission, yields downstream of production are accounted for by the production quantity
                           2: 1 / (stage3 * stage4 * stage5 * stage6 * stage7 * stage8 * stage9 * stage10),
                           3: 1 / (stage4 * stage5 * stage6 * stage7 * stage8 * stage9 * stage10),
                           4: 1 / (stage5 * stage6 * stage7 * stage8 * stage9 * stage10),
                           5: 1 / (stage6 * stage7 * stage8 * stage9 * stage10),
                           6: 1 / (stage7 * stage8 * stage9 * stage10),
                           7: 1 / (stage8 * stage9 * stage10),
                           8: 1 / (stage9 * stage10),
                           9: 1 / stage10,
                           10: 1}

        if self.pathwaytype == "Simple":
            transtype = self.tinfra[1].trans_type
            stage4 = self.storage_method[0].variables._yield * (
                        1 - self.storage_method[0].variables._loss_day * storage_days_by_transtype[transtype])
            stage5 = self.tinfra[1].variables._yield * (
                        1 - self.tinfra[1].variables._loss_km * self.pathwaydefinition[1].distance)
            self._yield = {1: 1 / stage1,
                           # primary energy transmission, yields downstream of production are accounted for by the production quantity
                           2: 1 / (stage5 * stage4 * stage3),
                           3: 1 / (stage4 * stage5),
                           4: 1 / stage5,
                           5: 1}
        n = 0
        # check if yields are not less than 0:
        for stage in [stage1, stage2, stage3, stage4, stage5]:
            if stage < 0:
                print('yield is less than 0', stage, n)
                exit()
            else:
                n + 1

    def calc_energy_bystage(self):
        # calculate the pathway energy intensity
        assigned_energies = self.assign_energysources()

        # same for simple and energy to prod
        stage2 = assigned_energies['Prod Fixed'].calc_energy(self.productionmeth, 1, self.energysource, self._yield[2])
        stage2_km = assigned_energies['Prod Dyn'].calc_energy(self.productionmeth, 1, self.energysource,
                                                              self.pathwaydefinition[1].distance * self._yield[2])
        stage3 = assigned_energies['Conv'].calc_energy(self.conversions[0], 1, self.energysource, self._yield[3])

        # energy to simple
        if self.pathwaytype == "Simple":
            stage4 = assigned_energies['Stor1'].calc_energy(self.storage_method[0], 1, self.energysource,
                                                            self._yield[4])
            stage5 = assigned_energies['T2 Fixed'].calc_energy(self.tinfra[1], 1, self.energysource, self._yield[5],
                                                               tinfracapacity=self.capacities[1])
            stage5_km = assigned_energies['T2 Dyn'].calc_energy(self.tinfra[1], 1, self.energysource,
                                                                self.pathwaydefinition[1].distance * self._yield[5],
                                                                tinfracapacity=self.capacities[1])
            combined_downstream = \
            combine_energydicts([stage2, stage2_km, stage3, stage4, stage5, stage5_km], self.energysource)['Total_ES']
            # calc upstrem energy required from primary energy source
            stage1 = assigned_energies['T1 Fixed'].calc_energy(self.tinfra[0], 1, self.energysource,
                                                               combined_downstream * self._yield[1],
                                                               tinfracapacity=self.capacities[0])
            stage1_km = assigned_energies['T1 Dyn'].calc_energy(self.tinfra[0], 1, self.energysource,
                                                                combined_downstream * self.pathwaydefinition[
                                                                    0].distance * self._yield[1],
                                                                tinfracapacity=self.capacities[0])
            energy_list = {'Stage 1': stage1, 'Stage 1 km': stage1_km, 'Stage 2': stage2, 'Stage 2 km': stage2_km,
                           'Stage 3': stage3, 'Stage 4': stage4,
                           'Stage 5': stage5, 'Stage 5 km': stage5_km}

        # energy to prod only
        if self.pathwaytype == "Energy to Prod":
            stage4 = assigned_energies['T2 Fixed'].calc_energy(self.tinfra[1], 1, self.energysource, self._yield[4],
                                                               tinfracapacity=self.capacities[1])
            stage4_km = assigned_energies['T2 Dyn'].calc_energy(self.tinfra[1], 1, self.energysource,
                                                                self.pathwaydefinition[1].distance * self._yield[4],
                                                                tinfracapacity=self.capacities[1])
            stage5 = assigned_energies['Conv2'].calc_energy(self.conversions[1], 1, self.energysource, self._yield[5])
            stage6 = assigned_energies['Stor1'].calc_energy(self.storage_method[0], 1, self.energysource,
                                                            self._yield[6])
            stage7 = assigned_energies['T3 Fixed'].calc_energy(self.tinfra[2], 1, self.energysource, self._yield[7],
                                                               tinfracapacity=self.capacities[2])
            stage7_km = assigned_energies['T3 Dyn'].calc_energy(self.tinfra[2], 1, self.energysource,
                                                                self.pathwaydefinition[2].distance * self._yield[7],
                                                                tinfracapacity=self.capacities[2])
            stage8 = assigned_energies['Conv3'].calc_energy(self.conversions[2], 1, self.energysource, self._yield[8])
            stage9 = assigned_energies['Stor1'].calc_energy(self.storage_method[1], 1, self.energysource,
                                                            self._yield[9])
            stage10 = assigned_energies['T4 Fixed'].calc_energy(self.tinfra[3], 1, self.energysource, self._yield[8],
                                                                tinfracapacity=self.capacities[3])
            stage10_km = assigned_energies['T4 Dyn'].calc_energy(self.tinfra[3], 1, self.energysource,
                                                                 self.pathwaydefinition[3].distance * self._yield[8],
                                                                 tinfracapacity=self.capacities[3])
            combined_downstream = combine_energydicts(
                [stage2, stage2_km, stage3, stage4, stage4_km, stage5, stage6, stage7, stage7_km, stage8, stage9,
                 stage10,
                 stage10_km], self.energysource)['Total_ES']
            # calc upstrem energy required from primary energy source
            stage1 = assigned_energies['T1 Fixed'].calc_energy(self.tinfra[0], 1, self.energysource,
                                                               combined_downstream * self._yield[1],
                                                               tinfracapacity=self.capacities[0])
            stage1_km = assigned_energies['T1 Dyn'].calc_energy(self.tinfra[0], 1, self.energysource,
                                                                combined_downstream * self.pathwaydefinition[
                                                                    0].distance * self._yield[1],
                                                                tinfracapacity=self.capacities[0])

            energy_list = {'Stage 1': stage1, 'Stage 1 km': stage1_km, 'Stage 2': stage2, 'Stage 2 km': stage2_km,
                           'Stage 3': stage3, 'Stage 4': stage4, 'Stage 4 km': stage4_km,
                           'Stage 5': stage5, 'Stage 6': stage6, 'Stage 7': stage7, 'Stage 7 km': stage7_km,
                           'Stage 8': stage8, 'Stage 9': stage9, 'Stage 10': stage10, 'Stage 10 km': stage10_km}

        list_stages = ['Stage 1', 'Stage 1 km', 'Stage 2', 'Stage 2 km', 'Stage 3', 'Stage 3 km', 'Stage 4',
                       'Stage 4 km', 'Stage 5', 'Stage 5 km',
                       'Stage 6', 'Stage 6 km', 'Stage 7', 'Stage 7 km', 'Stage 8', 'Stage 8 km', 'Stage 9',
                       'Stage 9 km', 'Stage 10', 'Stage 10 km']

        for stage in list_stages:
            if stage not in energy_list:
                energy_list[stage] = {"Embodied": {"Undefined": 0, "Total": 0}, "O_M": {"Undefined": 0, "Total": 0},
                                      "Process": {"Undefined": 0, "Total": 0}, "Total": 0}

        self.energy = energy_list
        self.energy_total = combine_energydicts(energy_list, self.energysource)

    def calc_emissionvalues(self, emissionsfactors, energies):

        if self.pathwaytype == 'Simple':
            combined_downstream_en = combine_energydicts(
                [energies['Stage 2'], energies['Stage 2 km'], energies['Stage 3'], energies['Stage 4'],
                 energies['Stage 4 km'], energies['Stage 5'], energies['Stage 5 km']], self.energysource)['Total_ES']
            stage1 = self.tinfra[0].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 1'],
                                                                             self.tinfra[0], 1,
                                                                             combined_downstream_en * self._yield[1],
                                                                             tinfracapacity=self.capacities[0])
            stage1_km = self.tinfra[0].variables.emissions_km.calc_emissions(emissionsfactors, energies['Stage 1 km'],
                                                                             self.tinfra[0], 1, combined_downstream_en *
                                                                             self.pathwaydefinition[0].distance *
                                                                             self._yield[1],
                                                                             tinfracapacity=self.capacities[0])
            stage2 = self.productionmeth.variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 2'],
                                                                                  self.productionmeth, 1,
                                                                                  self._yield[2])
            stage2_km = self.productionmeth.variables.emissions_km.calc_emissions(emissionsfactors,
                                                                                  energies['Stage 2 km'],
                                                                                  self.productionmeth, 1,
                                                                                  self.pathwaydefinition[1].distance *
                                                                                  self._yield[2])
            stage3 = self.conversions[0].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 3'],
                                                                                  self.conversions[0], 1,
                                                                                  self._yield[3])

            stage4 = self.storage_method[0].variables.emissions_fixed.calc_emissions(emissionsfactors,
                                                                                     energies['Stage 4'],
                                                                                     self.storage_method[0], 1,
                                                                                     self._yield[4])

            stage5 = self.tinfra[1].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 5'],
                                                                             self.tinfra[1], 1, self._yield[5],
                                                                             tinfracapacity=self.capacities[1])
            stage5_km = self.tinfra[1].variables.emissions_km.calc_emissions(emissionsfactors, energies['Stage 5 km'],
                                                                             self.tinfra[1], 1,
                                                                             self.pathwaydefinition[1].distance *
                                                                             self._yield[5],
                                                                             tinfracapacity=self.capacities[1])

            dict_emissions = {'Stage 1': stage1, 'Stage 1 km': stage1_km, 'Stage 2': stage2, 'Stage 2 km': stage2_km,
                              'Stage 3': stage3, 'Stage 4': stage4, 'Stage 5': stage5, 'Stage 5 km': stage5_km}

        if self.pathwaytype == 'Energy to Prod':
            combined_downstream_en = combine_energydicts(
                [energies['Stage 2'], energies['Stage 2 km'], energies['Stage 3'], energies['Stage 4'],
                 energies['Stage 4 km'], energies['Stage 5'], energies['Stage 5 km'], energies['Stage 6'],
                 energies['Stage 6 km'], energies['Stage 7'], energies['Stage 7 km'], energies['Stage 8'],
                 energies['Stage 9'], energies['Stage 10'], energies['Stage 10 km']], self.energysource)['Total_ES']

            stage1 = self.tinfra[0].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 1'],
                                                                             self.tinfra[0], 1,
                                                                             combined_downstream_en * self._yield[1],
                                                                             tinfracapacity=self.capacities[0])
            stage1_km = self.tinfra[0].variables.emissions_km.calc_emissions(emissionsfactors, energies['Stage 1 km'],
                                                                             self.tinfra[0], 1, combined_downstream_en *
                                                                             self.pathwaydefinition[0].distance *
                                                                             self._yield[1],
                                                                             tinfracapacity=self.capacities[0])
            stage2 = self.productionmeth.variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 2'],
                                                                                  self.productionmeth, 1,
                                                                                  self._yield[2])
            stage2_km = self.productionmeth.variables.emissions_km.calc_emissions(emissionsfactors,
                                                                                  energies['Stage 2 km'],
                                                                                  self.productionmeth, 1,
                                                                                  self.pathwaydefinition[1].distance *
                                                                                  self._yield[2])
            stage3 = self.conversions[0].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 3'],
                                                                                  self.conversions[0], 1,
                                                                                  self._yield[3])

            stage4 = self.tinfra[1].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 4'],
                                                                             self.tinfra[1], 1, self._yield[4],
                                                                             tinfracapacity=self.capacities[1])
            stage4_km = self.tinfra[1].variables.emissions_km.calc_emissions(emissionsfactors, energies['Stage 4 km'],
                                                                             self.tinfra[1], 1,
                                                                             self.pathwaydefinition[1].distance *
                                                                             self._yield[4],
                                                                             tinfracapacity=self.capacities[1])
            stage5 = self.conversions[0].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 5'],
                                                                                  self.conversions[1], 1,
                                                                                  self._yield[5])

            stage6 = self.storage_method[0].variables.emissions_fixed.calc_emissions(emissionsfactors,
                                                                                     energies['Stage 6'],
                                                                                     self.storage_method[0], 1,
                                                                                     self._yield[6])

            stage7 = self.tinfra[2].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 7'],
                                                                             self.tinfra[2], 1, self._yield[6],
                                                                             tinfracapacity=self.capacities[2])

            stage7_km = self.tinfra[2].variables.emissions_km.calc_emissions(emissionsfactors, energies['Stage 7 km'],
                                                                             self.tinfra[2], 1,
                                                                             self.pathwaydefinition[2].distance *
                                                                             self._yield[6],
                                                                             tinfracapacity=self.capacities[2])

            stage8 = self.conversions[2].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 8'],
                                                                                  self.conversions[2], 1,
                                                                                  self._yield[7])

            stage9 = self.storage_method[1].variables.emissions_fixed.calc_emissions(emissionsfactors,
                                                                                     energies['Stage 9'],
                                                                                     self.storage_method[1], 1,
                                                                                     self._yield[9])

            stage10 = self.tinfra[3].variables.emissions_fixed.calc_emissions(emissionsfactors, energies['Stage 10'],
                                                                              self.tinfra[3], 1, self._yield[8],
                                                                              tinfracapacity=self.capacities[3])

            stage10_km = self.tinfra[3].variables.emissions_km.calc_emissions(emissionsfactors, energies['Stage 10 km'],
                                                                              self.tinfra[3], 1,
                                                                              self.pathwaydefinition[3].distance *
                                                                              self._yield[8],
                                                                              tinfracapacity=self.capacities[3])
            if stage10['Embodied'] > 1:
                print(stage10['Embodied'], self.tinfra[3].ID, self.capacities[3], self._yield[8])

            dict_emissions = {'Stage 1': stage1, 'Stage 1 km': stage1_km, 'Stage 2': stage2, 'Stage 2 km': stage2_km,
                              'Stage 3': stage3, 'Stage 4': stage4, 'Stage 4 km': stage4_km,
                              'Stage 5': stage5, 'Stage 6': stage6, 'Stage 7': stage7, 'Stage 7 km': stage7_km,
                              'Stage 8': stage8, 'Stage 9': stage9, 'Stage 10': stage10, 'Stage 10 km': stage10_km}

        list_stages = ['Stage 1', 'Stage 1 km', 'Stage 2', 'Stage 2 km', 'Stage 3', 'Stage 3 km', 'Stage 4',
                       'Stage 4 km', 'Stage 5', 'Stage 5 km',
                       'Stage 6', 'Stage 6 km', 'Stage 7', 'Stage 7 km', 'Stage 8', 'Stage 8 km', 'Stage 9',
                       'Stage 9 km', 'Stage 10', 'Stage 10 km']

        for stage in list_stages:
            if stage not in dict_emissions.keys():
                dict_emissions[stage] = {"Embodied": 0, "O_M": 0, "Process": 0, "Total": 0}

        emissions_total = combine_emissionsdicts(dict_emissions)
        self.emissions = dict_emissions
        self.emissions_total = emissions_total

        return dict_emissions, emissions_total

    def calc_values(self, x):
        self.calc_yield()
        self.calc_energy_bystage()
        if self.energy_total['Total'] < 0:
            print(self.pathwaydefinition[0].origin.ID, self.pathwaydefinition[0].destination.ID,
                  self.pathwaydefinition[0].destination.country, self.energy_total)
            exit()
        self.calc_emissionvalues(energysources.instances, self.energy)
        num = self.pathwaynum
        for use in self.end_uses:
            result = results(str(x), num, self.pathwaydefinition, self.pathwaytype, use, self.productionmeth,
                             self.vectors, self.tinfra, self.energysource, self.energy_total,
                             self.emissions_total, self.energy, self.emissions, self._yield,
                             self.energysource.emissionsfactor)
        return self.emissions_total['Total']

    def sobol(self):
        self.production_utilisation()
        self.calc_yield()
        self.calc_energy_bystage()
        self.calc_emissionvalues(energysources.instances, self.energy)
        return self.emissions_total['Total']


class transmission_paths():
    instances = {}

    def __init__(self, origin, destination, onshoreoroffshore, typetransmission, pathwaystage):
        self.origin = origin
        self.destination = destination
        self.typetransmission = typetransmission
        self.onshoreoroffshore = onshoreoroffshore
        self.pathwaystage = pathwaystage
        self.vectors = None
        self.distance = None
        self.instances[self.origin.ID, self.destination.ID] = self
        self.get_distance()
        self.possible_vectors()

    def check_for_duplicates(dictroutes):
        length_dict = len(dictroutes)
        length_unique = len(set(dictroutes.keys()))
        if length_dict != length_unique:
            print("duplicates exist")

    def get_distance(self):
        # check if it has been defined as onshore or offshore, note geopy and searoute take lat,long in opposite order
        if self.onshoreoroffshore == "Offshore":
            origin_coords = [self.origin.long, self.origin.lat]
            destination_coords = [self.destination.long, self.destination.lat]
            route = sr.searoute(origin_coords, destination_coords, units="km")
            self.distance = route.properties['length']
        elif self.onshoreoroffshore == "Onshore":
            origin_coords = [self.origin.lat, self.origin.long]
            destination_coords = [self.destination.lat, self.destination.long]
            self.distance = distance.distance(origin_coords, destination_coords).km
        else:
            print("Error - No Distance Found")
            self.distance = "NA"

    def possible_vectors(self):
        vector_list = []
        if self.typetransmission == "Energy to Prod":
            vector_list.append(self.origin.energytype)
        elif self.typetransmission == "Energy to Port":
            # confirmed possible when created
            vector_list.append(self.origin.energytype)
        else:
            for vector in self.origin.vectors:
                if vector in self.destination.vectors:
                    vector_list.append(vector)
        self.vectors = vector_list


class Transmission_infra():
    # transmission infrastructure eg cable, pipeline, tanker, does not include the conversion from one vector to another
    instances = {}

    def __init__(self, ID, processname, vector, capacity, unit, onshoreoffshore, transtype, distance_lim,
                 variables=None):
        self.ID = ID
        self.processname = processname
        self.vector = vector
        self.capacity = capacity
        self.unit = unit
        self.onshoreoffshore = onshoreoffshore
        self.trans_type = transtype
        self.variables = variables
        self.distance_lim = distance_lim
        self.instances[self.ID] = self

    def add_variables(self, dataframe):
        df = dataframe.loc[dataframe['ID'] == self.ID]
        energy_km = energy(convert_to_dict(df['Energy_embodiedkm'].values[0]),
                           convert_to_dict(df['Energy_processkm'].values[0]),
                           convert_to_dict(df['Energy_OMkm'].values[0]))
        energy_fixed = energy(convert_to_dict(df['Energy_embodied'].values[0]),
                              convert_to_dict(df['Energy_process'].values[0]),
                              convert_to_dict(df['Energy_OM'].values[0]))
        emissions_km = emissions(mc(df['Emissions_embodiedkm'].values[0]), mc(df['Emissions_processkm'].values[0]),
                                 mc(df['Emissions_OMkm'].values[0]), mc(df['H2emissionskm'].values[0]))
        emissions_fixed = emissions(mc(df['Emissions_embodied'].values[0]), mc(df['Emissions_process'].values[0]),
                                    mc(df['Emissions_OM'].values[0]), mc(df['H2emissions'].values[0]))

        self.variables = mc_variables(df['Lifetime'].values[0], df['Capacity Factor'].values[0], df['Yield'].values[0],
                                      energy_fixed, emissions_fixed, emissions_km, energy_km, df['_loss_km'].values[0],
                                      df['Speed'].values[0])
        self.capacity = df['Capacity'].values[0] * self.variables.capacity_factor
        self.instances[self.ID] = self

    def sobol_update(self, _yield, capacityfactor, _loss_km, lifetime, speed, emissionsembodied, emissionsembodiedkm,
                     emissionsprocess, emissionsprocesskm, emissionsOM, emissionsOMkm, energyembodied, energyembodiedkm,
                     energyprocess, energyprocesskm, energyOM, energyOMkm, h2emissions, h2emissionskm):
        capacity = self.capacity / self.variables.capacity_factor
        self.variables._yield = _yield[0]
        self.variables.capacity_factor = capacityfactor[0]
        self.capacity = capacity * self.variables.capacity_factor
        self.variables._loss_km = _loss_km[0]
        self.variables.lifetime = lifetime[0]
        self.variables.speed = speed[0]
        self.variables.energy_km = energy(sobol_add_undefined(energyembodiedkm), sobol_add_undefined(energyprocesskm),
                                          sobol_add_undefined(energyOMkm))
        self.variables.energy_fixed = energy(sobol_add_undefined(energyembodied), sobol_add_undefined(energyprocess),
                                             sobol_add_undefined(energyOM))
        self.variables.emissions_km = emissions(emissionsembodiedkm[0], emissionsprocesskm[0], emissionsOMkm[0],
                                                h2emissionskm[0])
        self.variables.emissions_fixed = emissions(emissionsembodied[0], emissionsprocess[0], emissionsOM[0],
                                                   h2emissions[0])
        self.instances[self.ID] = self


class mc_variables():

    def __init__(self, lifetime, capacity_factor, _yield, energy_fixed, emissions_fixed, emissions_km=None,
                 energy_km=None, _loss_km=None, speed=None, _loss_day=None):
        self.lifetime = mc(lifetime)
        self.capacity_factor = mc(capacity_factor)
        self._yield = mc(_yield)
        self.energy_fixed = energy_fixed
        self.emissions_fixed = emissions_fixed
        self.energy_km = energy_km
        self.emissions_km = emissions_km
        self._loss_day = mc(_loss_day)
        self.speed = mc(speed)
        self._loss_km = mc(_loss_km)


class H2_production():
    # hydrogen production method
    instances = {}

    def __init__(self, ID, location, scale, capacity, e_type, distance_offshore=0, variables=None, other_inputs=None):
        self.ID = ID
        self.scale = scale
        self.location = location
        self.capacity = capacity
        self.distance_offshore = distance_offshore
        self.e_type = e_type
        self.other_inputs = other_inputs
        self.variables = variables
        self.instances[self.ID] = self

    def add_variables(self, dataframe):
        df = dataframe.loc[dataframe['ID'] == self.ID]
        energy_km = energy(convert_to_dict(df['Energy_embodiedkm'].values[0]),
                           convert_to_dict(df['Energy_processkm'].values[0]),
                           convert_to_dict(df['Energy_OMkm'].values[0]))
        energy_fixed = energy(convert_to_dict(df['Energy_embodied'].values[0]),
                              convert_to_dict(df['Energy_process'].values[0]),
                              convert_to_dict(df['Energy_OM'].values[0]))
        emissions_km = emissions(mc(df['Emissions_embodiedkm'].values[0]), mc(df['Emissions_processkm'].values[0]),
                                 mc(df['Emissions_OMkm'].values[0]), mc(df['H2emissionskm'].values[0]))
        emissions_fixed = emissions(mc(df['Emissions_embodied'].values[0]), mc(df['Emissions_process'].values[0]),
                                    mc(df['Emissions_OM'].values[0]), mc(df['H2emissions'].values[0]))

        self.variables = mc_variables(df['Lifetime'].values[0], df['Capacity Factor'].values[0], df['_yield'].values[0],
                                      energy_fixed, emissions_fixed, emissions_km, energy_km, df['_loss_km'].values[0])
        self.instances[self.ID] = self

    def sobol_update(self, _yield, capacityfactor, lifetime, emissionsembodied, emissionsprocess, emissionsOM,
                     energyembodied, energyprocess, energyOM, h2emissions):
        capacity = self.capacity / self.variables.capacity_factor
        self.variables._yield = _yield[0]
        self.variables.capacity_factor = capacityfactor[0]
        self.capacity = capacity * self.variables.capacity_factor
        self.variables.lifetime = lifetime[0]
        self.variables.energy_fixed = energy(sobol_add_undefined(energyembodied), sobol_add_undefined(energyprocess),
                                             sobol_add_undefined(energyOM))
        self.variables.emissions_fixed = emissions(emissionsembodied[0], emissionsprocess[0], emissionsOM[0],
                                                   h2emissions[0])
        self.instances[self.ID] = self


class conversions():
    instances = {}

    def __init__(self, ID, inputvector, outputvector, capacity, variables=None):
        self.ID = ID
        self.capacity = capacity
        self.inputvector = inputvector
        self.outputvector = outputvector
        self.variables = variables
        self.instances[ID] = self

    def add_variables(self, dataframe):
        df = dataframe.loc[dataframe['ID'] == self.ID]
        energy_fixed = energy(convert_to_dict(df['Energy_embodied'].values[0]),
                              convert_to_dict(df['Energy_process'].values[0]),
                              convert_to_dict(df['Energy_OM'].values[0]))
        emissions_fixed = emissions(mc(df['Emissions_embodied'].values[0]), mc(df['Emissions_process'].values[0]),
                                    mc(df['Emissions_OM'].values[0]), mc(df['H2emissions'].values[0]))
        self.variables = mc_variables(df['Lifetime'].values[0], df['Capacity Factor'].values[0], df['Yield'].values[0],
                                      energy_fixed, emissions_fixed)
        self.capacity = df['Capacity'].values[0] * self.variables.capacity_factor

    def sobol_update(self, _yield, capacityfactor, lifetime, emissionsembodied, emissionsprocess, emissionsOM,
                     energyembodied, energyprocess, energyOM, h2emissions):
        capacity = self.capacity / self.variables.capacity_factor
        self.variables._yield = _yield[0]
        self.variables.capacity_factor = capacityfactor[0]
        self.capacity = capacity * self.variables.capacity_factor
        self.variables.lifetime = lifetime[0]
        self.variables.energy_fixed = energy(sobol_add_undefined(energyembodied), sobol_add_undefined(energyprocess),
                                             sobol_add_undefined(energyOM))
        self.variables.emissions_fixed = emissions(emissionsembodied[0], emissionsprocess[0], emissionsOM[0],
                                                   h2emissions[0])
        self.instances[self.ID] = self


class energysources():
    instances = {}

    def __init__(self, ID, energytype, emissions, utilisationfactor):
        self.ID = ID
        self.energytype = energytype
        self.emissionsfactor = emissions
        self.utilisationfactor = utilisationfactor
        self.instances[ID] = self

    def add_variables(self, dataframe):
        df = dataframe.loc[dataframe['ID'] == self.ID]
        self.utilisationfactor = mc(df['Utilisation Factor'].values[0])
        self.emissionsfactor = mc(df['Emissions (gCO2e/kWh)'].values[0]) / 1000
        self.instances[self.ID] = self

    def sobol_update(self, emissionsfactor, utilisationfactor):
        self.utilisationfactor = utilisationfactor[0]
        self.emissionsfactor = emissionsfactor[0] / 1000
        self.instances[self.ID] = self


class uses():
    instances = {}

    def __init__(self, ID, baseemissions, electrificationemissions, unit, vectors, energy, otheremissions, input, minh2,
                 totaldemand, H2emissions=0):
        self.ID = ID
        self.baseemissions = baseemissions
        self.electrificationemissions = electrificationemissions
        self.unit = unit
        self.vector = vectors
        self.energy = energy
        self.emissions = otheremissions
        self.H2emissions = H2emissions
        self.minh2 = minh2
        self.totaldemand = totaldemand
        self.input = input
        self.instances[ID] = self


class storage():
    instances = {}

    def __init__(self, ID, location, vector, capacity, variables=None):
        self.ID = ID
        self.vector = vector
        self.location = location
        self.capacity = capacity
        self.variables = variables
        self.instances[ID] = self

    def add_variables(self, dataframe):
        df = dataframe.loc[dataframe['ID'] == self.ID]
        energy_fixed = energy(convert_to_dict(df['Energy_embodied'].values[0]),
                              convert_to_dict(df['Energy_process'].values[0]),
                              convert_to_dict(df['Energy_OM'].values[0]))
        emissions_fixed = emissions(mc(df['Emissions_embodied'].values[0]), mc(df['Emissions_process'].values[0]),
                                    mc(df['Emissions_OM'].values[0]), mc(df['H2emissions'].values[0]))

        self.variables = mc_variables(df['Lifetime'].values[0], df['Capacity Factor'].values[0], df['Yield'].values[0],
                                      energy_fixed, emissions_fixed, df['H2emissions'].values[0],
                                      _loss_day=df['_loss_day'].values[0])
        self.capacity = df['Capacity'].values[0] * self.variables.capacity_factor
        self.instances[self.ID] = self

    def sobol_update(self, _yield, capacityfactor, _loss_day, lifetime, emissionsembodied, emissionsprocess,
                     emissionsOM, energyembodied, energyprocess,
                     energyOM, h2emissions):
        capacity = self.capacity / self.variables.capacity_factor
        self.variables._yield = _yield[0]
        self.variables.capacity_factor = capacityfactor[0]
        self.capacity = capacity * self.variables.capacity_factor
        self.variables.lifetime = lifetime[0]
        self.variables.energy_fixed = energy(sobol_add_undefined(energyembodied), sobol_add_undefined(energyprocess),
                                             sobol_add_undefined(energyOM))
        self.variables.emissions_fixed = emissions(emissionsembodied[0], emissionsprocess[0], emissionsOM[0],
                                                   h2emissions[0])
        self.variables._loss_day = _loss_day[0]
        self.instances[self.ID] = self


"""Define Functions to Input Data"""


def dropna(df):
    df = df.dropna(how="all", inplace=True)


def converttolist(value):
    # print(value)
    if pd.isnull(value) or value == "":
        uses = []
    else:
        uses = value.split(',')
        # print(uses)
    for i in range(len(uses)):
        uses[i] = uses[i].replace(" ", "")
        # print(uses)
    return uses


def sobol_add_undefined(value):
    if type(value) == list:
        return {"Undefined": value[0]}
    else:
        return value


def convert_to_dict(value, MC=True):
    if pd.isnull(value) or value == "":
        return {"Undefined": 0}
    if type(value) == int or type(value) == float:
        return {"Undefined": value}
    try:
        value.item()
        if isinstance(value.item(), int) or isinstance(value.item(), float):
            return {"Undefined": value}
        else:
            dict = {}
            for pair in value.split(','):
                dictpair = pair.split(':')
                dict[dictpair[0]] = dictpair[1].split(';')
                if len(dict[dictpair[0]]) == 1:
                    dict[dictpair[0]] = float(dict[dictpair[0]][0])
                if type(dict[dictpair[0]]) == list and MC == True:
                    low = float(dict[dictpair[0]][0])
                    high = float(dict[dictpair[0]][1])
                    dist_type = dict[dictpair[0]][2]
                    # assume all uniform distributions for now
                    mc_value = np.random.uniform(low, high)
                    dict[dictpair[0]] = mc_value
    except AttributeError:
        dict = {}
        for pair in value.split(','):
            dictpair = pair.split(':')
            dict[dictpair[0]] = dictpair[1].split(';')
            if len(dict[dictpair[0]]) == 1:
                dict[dictpair[0]] = float(dict[dictpair[0]][0])
            if type(dict[dictpair[0]]) == list and MC == True:
                low = float(dict[dictpair[0]][0])
                high = float(dict[dictpair[0]][1])
                # dist_type = dict[dictpair[0]][2]
                # assume all uniform distributions for now
                mc_value = np.random.uniform(low, high)
                dict[dictpair[0]] = mc_value
        return dict


def mc(value, MC=True):
    if pd.isnull(value) or value == "":
        return 0
    if type(value) == int or type(value) == float:
        if value < 0:
            print(value)
        return value
    try:
        value.item()
        if isinstance(value.item(), int) or isinstance(value.item(), float):
            return value
        else:
            value_list = value.split(';')
            if type(value_list) == list and MC == True:
                low = float(value_list[0])
                high = float(value_list[1])
                # dist_type = value_list[2]
                # assume all uniform distributions for now
                mc_value = np.random.uniform(low, high)
    except AttributeError:
        # print(value)
        value_list = value.split(';')
        if type(value_list) == list and MC == True:
            low = float(value_list[0])
            high = float(value_list[1])
            # dist_type = value_list[2]
            # assume all uniform distributions for now
            mc_value = np.random.uniform(low, high)
    return mc_value


def add_locations(dataframe):
    dropna(dataframe)
    dataframe.replace({np.nan: None}, inplace=True)
    for index, row in dataframe.iterrows():
        prodmethods = []
        if row.Electrolysis == "y":
            prodmethods.append('E')
        if row.ATR == "y":
            prodmethods.append('ATR')
        list_vectors = converttolist(row.Vectors)
        list_uses = converttolist(row.Use)
        list_energysources = converttolist(row.EnergySources)
        list_energytypes = converttolist(row.EnergyTypes)
        locations(row.ID, row['Location Type'], row.Location, row.Country,
                  row.Lat, row.Long, prodmethods, list_vectors, list_energytypes, list_energysources, list_uses,
                  row['Max Supply (ktH2/yr)'])

    return locations.instances


def add_transinfra(dataframe):
    for index, row in dataframe.iterrows():
        a = Transmission_infra(row.ID, row['Process Name'], row.Vector, row.Capacity, row.Unit, row.Location,
                               row.TransType, row.Distance_Lim)
        a.add_variables(dataframe)
    return Transmission_infra.instances


def add_production(dataframe):
    for index, row in dataframe.iterrows():
        a = H2_production(row.ID, row.Location, row.Scale, row.Capacity, row.E_type)
        a.add_variables(dataframe)
    return H2_production.instances


def add_conversions(dataframe):
    for index, row in dataframe.iterrows():
        a = conversions(row.ID, row.InputVector, row.OutputVector, row.Capacity)
        a.add_variables(dataframe)
    return conversions.instances


def add_energysources(dataframe):
    for index, row in dataframe.iterrows():
        a = energysources(row.ID, row.E_type, mc(row['Emissions (gCO2e/kWh)']) / 1000, mc(row['Utilisation Factor']))
    return energysources.instances


def add_uses(dataframe):
    for index, row in dataframe.iterrows():
        a = uses(row.ID, row['Base emissions (kgCO2e/unit)'], row['Electrification Emissions (kgCO2e/unit)'], row.Unit,
                 row.Vector, row['Energy Inputs (kWh)'],
                 row['Direct Emissions'], row['Input (kgH2e/unit)'], row['Min Hydrogen (kg/yr)'],
                 row['Total Demand (kgH2/yr)'])
    return uses.instances


def add_storage(dataframe):
    for index, row in dataframe.iterrows():
        a = storage(row.ID, row.Location, row.Vector, row['Capacity'])
        a.add_variables(dataframe)
    return storage.instances


"""Define Functions to Create Transmission Paths"""


def _same_country(location1, location2):
    if location1.country in (location2.country, CountryInfo(location2.country).alt_spellings(),
                             CountryInfo(location2.country).info()['name'], CountryInfo(location2.country).borders()):
        return True
    else:
        return False


def _create_trans_stage(df_origins, df_destinations, samecountry_TF, locations_class, onshoreoffshore):
    # list possible options given list of origins and destinations, used in createalltrans
    options = []
    for index_orig, row_orig in df_origins.iterrows():
        location1 = locations_class[row_orig.ID]
        for index_dest, row_dest in df_destinations.iterrows():
            location2 = locations_class[row_dest.ID]
            if _same_country(location1, location2) == samecountry_TF:
                options.append([location1, location2, onshoreoffshore])
            elif samecountry_TF == True and onshoreoffshore == "Onshore" and CountryInfo(location2.country).iso(
                    3) in CountryInfo(location1.country).borders():
                options.append([location1, location2, onshoreoffshore])

    return options


def createalltrans_stages(locations_df, locations_class, use_country_listnames):
    # create the transmission stages (e.g. Offshore wind to hydrogen production in teesside)

    ports = locations_df.loc[locations_df['Location Type'] == "Port"]
    portsinternational = ports.loc[~ports['Country'].isin(use_country_listnames)]
    portscountry = ports.loc[ports['Country'].isin(use_country_listnames)]
    energylocs = locations_df.loc[locations_df['Location Type'] == "Energy"]
    prodlocs = locations_df.loc[locations_df['Location Type'] == "Production"]
    uselocs = locations_df.loc[locations_df['Location Type'] == "Uses"]
    print(len(energylocs), len(prodlocs), len(uselocs), len(portsinternational), len(portscountry))

    """Add prod - port"""
    for option in _create_trans_stage(prodlocs, portsinternational, True, locations_class, "Onshore"):
        transmission_paths(option[0], option[1], option[2], "Prod to Port", 0)

    """Add UK port - Use"""
    for option in _create_trans_stage(portscountry, uselocs, True, locations_class, "Onshore"):
        transmission_paths(option[0], option[1], option[2], "Port to Use", 0)

    """Port to UK port"""
    for option in _create_trans_stage(portsinternational, portscountry, False, locations_class, "Offshore"):
        transmission_paths(option[0], option[1], option[2], "Port to Port", 0)

    """Add energy to production or port stages"""
    for index_en, row_en in energylocs.iterrows():
        location1 = locations_class[row_en.ID]
        for index_prod, row_prod in prodlocs.iterrows():
            location2 = locations_class[row_prod.ID]
            if _same_country(location1, location2) == True:
                if location1.offshoreonshore not in (location2.offshoreonshore,
                                                     "Onshore") or location2.offshoreonshore == "Offshore":
                    onshoreoffshore = "Offshore"
                else:
                    onshoreoffshore = "Onshore"

                if "E" in row_en.EnergyTypes and row_prod.Electrolysis == "y":
                    # print(row_en.EnergyTypes)
                    transmission_paths(location1, location2, onshoreoffshore, "Energy to Prod", 0)

                elif "NG" in row_en.EnergyTypes and row_prod.ATR == "y":
                    transmission_paths(location1, location2, onshoreoffshore, "Energy to Prod", 0)
        for index_port, row_port in portsinternational.iterrows():
            location2 = locations_class[row_port.ID]
            if _same_country(location1, location2) == True and row_en.EnergyTypes in row_port.Vectors:
                if location1.offshoreonshore not in (location2.offshoreonshore,
                                                     "Onshore") or location2.offshoreonshore == "Offshore":
                    onshoreoffshore = "Offshore"
                else:
                    onshoreoffshore = "Onshore"
                transmission_paths(location1, location2, onshoreoffshore, "Energy to Port", 0)

    """Add UK prod - use"""
    for option in _create_trans_stage(prodlocs, uselocs, True, locations_class, "Onshore"):
        transmission_paths(option[0], option[1], option[2], "Prod to Use", 0)

    """Add port - prod"""
    for option in _create_trans_stage(portscountry, prodlocs, True, locations_class, "Onshore"):
        transmission_paths(option[0], option[1], option[2], "Port to Prod", 0)

    return transmission_paths.instances


def connect_transpaths_to_supplychain(transmission_paths):
    # make dictionary of possible transmission stages by type (energy to prod, prod to use etc.)
    transmission_dict = {}
    for stage in transmission_paths.values():
        typetransmission = stage.typetransmission
        if typetransmission not in transmission_dict:
            transmission_dict[typetransmission] = []
        transmission_dict[typetransmission].append(stage)
    # dictionary to hold the three types of supply chain, 1 = local (energy-prod-use), 2 = energy-prod-port-port-use, 3 = energy-port-port-prod-use
    H2_supplychains = {1: [], 2: [], 3: []}

    # Class 1 pathways: energy to prod to use, including the vector options
    for stage in transmission_dict.get('Energy to Prod', []):
        # print(stage.origin.ID)
        for vector1 in stage.vectors:
            # print(vector1)
            for stage2 in transmission_dict.get("Prod to Use", []):

                for vector2 in stage2.vectors:

                    if stage.destination.ID == stage2.origin.ID:
                        H2_supplychains[1].append([stage, stage2, vector1, vector2])
    # print(len(H2_supplychains[1]))

    # Class 2 pathways: energy to prod to port to port to use, including vector options, but preventing both NH3 and Methanol use in same chain
    for stage in transmission_dict.get('Energy to Prod', []):
        for vector1 in stage.vectors:
            for stage2 in transmission_dict.get("Prod to Port", []):
                for vector2 in stage2.vectors:
                    if stage.destination.ID == stage2.origin.ID:
                        for stage3 in transmission_dict.get("Port to Port", []):
                            for vector3 in stage3.vectors:
                                if vector3 in ('NG', 'E'):
                                    pass
                                else:
                                    if vector3 == 'CH2' and stage3.distance > 1000:
                                        pass
                                    else:
                                        if stage2.destination.ID == stage3.origin.ID:
                                            for stage4 in transmission_dict.get("Port to Use", []):
                                                for vector4 in stage4.vectors:
                                                    if vector4 in ('NG', 'E'):
                                                        pass
                                                    else:
                                                        if stage3.destination.ID == stage4.origin.ID:
                                                            # prevent both NH3 and Methanol use in same chain, as well as switching from NH3 or CH3OH to hydrogen and back
                                                            if 'NH3' in [vector1, vector2, vector3,
                                                                         vector4] and 'CH3OH' in [vector1, vector2,
                                                                                                  vector3, vector4]:
                                                                pass
                                                            elif vector2 != vector3 and vector4 not in (vector3, 'CH2',
                                                                                                        'LH2'):
                                                                pass
                                                            else:
                                                                H2_supplychains[2].append(
                                                                    [stage, stage2, stage3, stage4, vector1, vector2,
                                                                     vector3, vector4])

    """"
    #Class 3 pathways: energy to port to port to prod to use
    for stage in transmission_dict.get('Energy to Port', []):
        for vector1 in stage.vectors:
            for stage2 in transmission_dict.get('Port to Port', []):
                for vector2 in stage2.vectors:
                    if stage.destination.ID == stage2.origin.ID and vector2 == vector1:
                        for stage3 in transmission_dict.get('Port to Prod', []):
                                #assume E or NG can be transferred from the port to the production site
                                if stage2.destination.ID == stage3.origin.ID:
                                    for stage4 in transmission_dict.get('Prod to Use', []):
                                        for vector4 in stage4.vectors:
                                            if stage3.destination.ID == stage4.origin.ID:
                                                #only vector 4 is hydrogen/hydrogen carrier so no further limitations on which it can be
                                                H2_supplychains[3].append([stage,stage2,stage3,stage4,vector1,vector2,vector2,vector4])"""

    return H2_supplychains


def add_transinfra_andprodmethod_to_supplychain(H2_supplychains, t_infra_df, tinfra_class, storage_df):
    H2supplyoptions = []
    print('supplychains', len(H2_supplychains[1]), len(H2_supplychains[2]), len(H2_supplychains[3]))
    storage_type = {"Tanker": "Tanker", "Pipeline": "Other", "Cable": "Other", "Truck": "Other"}

    # add transmission infrastructure to class 1 supply chains
    for supplychain in H2_supplychains[1]:
        for production in H2_production.instances.values():
            if production.e_type in supplychain[0].origin.energytype:
                for index, row in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[2][0]) & (
                        t_infra_df['Location'] == supplychain[0].onshoreoroffshore) &
                                                 (t_infra_df['Distance_Lim'] > supplychain[0].distance)].iterrows():
                    tinfra = row.ID
                    for index2, row2 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[3]) & (
                            t_infra_df['Location'] == supplychain[1].onshoreoroffshore) &
                                                       (t_infra_df['Distance_Lim'] > supplychain[
                                                           1].distance)].iterrows():
                        tinfra2 = row2.ID
                        storage1 = storage.instances[
                            storage_df.loc[(storage_df['Vector'] == supplychain[3])].iloc[0].ID]
                        H2supplyoptions.append({'Pathway': [supplychain[0], supplychain[1]], 'Pathway Type': 'Simple',
                                                'Production Meth': production,
                                                'Vectors': [supplychain[2][0], supplychain[3]],
                                                'Transmission Infra': [tinfra_class[tinfra], tinfra_class[tinfra2]],
                                                'Storage': [storage1]})

                        # Add T infra to class 2 supply chains
    for supplychain in H2_supplychains[2]:

        for production in H2_production.instances.values():
            if production.e_type in supplychain[0].origin.energytype and production.scale in ("Centralised - PEM",
                                                                                              'Centralised NG'):
                for index, row in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[4][0]) & (
                        t_infra_df['Location'] == supplychain[0].onshoreoroffshore) &
                                                 (t_infra_df['Distance_Lim'] > supplychain[0].distance)].iterrows():
                    tinfra = row.ID
                    for index2, row2 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[5]) & (
                            t_infra_df['Location'] == supplychain[1].onshoreoroffshore) &
                                                       (t_infra_df['Distance_Lim'] > supplychain[
                                                           1].distance)].iterrows():
                        tinfra2 = row2.ID
                        for index3, row3 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[6]) & (
                                t_infra_df['Location'] == supplychain[2].onshoreoroffshore) &
                                                           (t_infra_df['Distance_Lim'] > supplychain[
                                                               2].distance)].iterrows():
                            tinfra3 = row3.ID
                            for index4, row4 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[7]) & (
                                    t_infra_df['Location'] == supplychain[3].onshoreoroffshore) &
                                                               (t_infra_df['Distance_Lim'] > supplychain[
                                                                   3].distance)].iterrows():
                                tinfra4 = row4.ID
                                storage1 = storage.instances[
                                    storage_df.loc[(storage_df['Vector'] == supplychain[5])].iloc[0].ID]
                                storage2 = storage.instances[
                                    storage_df.loc[(storage_df['Vector'] == supplychain[6])].iloc[0].ID]
                                H2supplyoptions.append(
                                    {'Pathway': [supplychain[0], supplychain[1], supplychain[2], supplychain[3]],
                                     'Pathway Type': 'Energy to Prod', 'Production Meth': production,
                                     'Vectors': [supplychain[4][0], supplychain[5], supplychain[6], supplychain[7]],
                                     'Transmission Infra': [tinfra_class[tinfra], tinfra_class[tinfra2],
                                                            tinfra_class[tinfra3], tinfra_class[tinfra4]],
                                     'Storage': [storage1, storage2]})

    # Add T infra to class 3 supply chain
    for supplychain in H2_supplychains[3]:
        for production in H2_production.instances.values():
            if production.e_type in supplychain[0].origin.energytype:
                for index, row in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[4]) & (
                        t_infra_df['Location'] == supplychain[0].onshoreoroffshore) &
                                                 (t_infra_df['Distance_Lim'] > supplychain[0].distance)].iterrows():
                    tinfra = row.ID
                    for index2, row2 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[5]) & (
                            t_infra_df['Location'] == supplychain[1].onshoreoroffshore) &
                                                       (t_infra_df['Distance_Lim'] > supplychain[
                                                           1].distance)].iterrows():
                        tinfra2 = row2.ID
                        for index3, row3 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[6]) & (
                                t_infra_df['Location'] == supplychain[2].onshoreoroffshore) &
                                                           (t_infra_df['Distance_Lim'] > supplychain[
                                                               2].distance)].iterrows():
                            tinfra3 = row3.ID
                            for index4, row4 in t_infra_df.loc[(t_infra_df['Vector'] == supplychain[7]) & (
                                    t_infra_df['Location'] == supplychain[3].onshoreoroffshore) &
                                                               (t_infra_df['Distance_Lim'] > supplychain[
                                                                   3].distance)].iterrows():
                                tinfra4 = row4.ID
                                storage1 = storage.instances[
                                    storage_df.loc[(storage_df['Vector'] == supplychain[7])].iloc[0].ID]
                                H2supplyoptions.append(
                                    {'Pathway': [supplychain[0], supplychain[1], supplychain[2], supplychain[3]],
                                     'Pathway Type': 'Energy to Port', 'Production Meth': production,
                                     'Vectors': [supplychain[4][0], supplychain[5], supplychain[6], supplychain[7]],
                                     'Transmission Infra': [tinfra_class[tinfra], tinfra_class[tinfra2],
                                                            tinfra_class[tinfra3], tinfra_class[tinfra4]],
                                     'Storage': []})
    print(len(H2supplyoptions))
    return H2supplyoptions


def add_energysources_to_supplychain(H2supplyoptions, energysources_df,
                                     nonprocessenergy=["Undefined", 'Shipping', 'ShipFuel', 'Deisel']):
    n = 1
    for x in H2supplyoptions:
        options = []
        # add energysources from that match the energy type and country
        for index, row in energysources_df.loc[(energysources_df['E_type'] == x['Production Meth'].e_type) & (
        energysources_df['ID'].isin(x['Pathway'][0].origin.energysources))].iterrows():
            if row.Availability in ('Primary') and row.Country not in ('Global'):
                options.append(energysources.instances[row.ID])

        # if options empty, add global options but these are not used if viable options have been found
        if len(options) < 1:
            for index, row in energysources_df.loc[(energysources_df['E_type'] == x['Production Meth'].e_type) & (
                    energysources_df['Country'] == "Global")].iterrows():
                if row.ID not in nonprocessenergy and row.Availability in ('Secondary', 'Both'):
                    options.append(energysources.instances[row.ID])

        # create supply chain with each option found as the energy source

        for option in options:
            # print(x['Pathway Type'])
            a = H2_supply_chain(n, x['Pathway'], x['Pathway Type'], x['Production Meth'], x['Vectors'],
                                x['Transmission Infra'], x['Storage'], option)
            n += 1
            # print(a.pathwaynum)
    print('Number of pathways: ', len(H2_supply_chain.instances))
    return H2_supply_chain.instances


def combine_energydicts(energyinputs, es):
    combined = {}
    total = 0
    total_es = 0

    if type(energyinputs) == dict:
        temp = energyinputs
        energyinputs = []
        for key, value in temp.items():
            energyinputs.append(value)

    for energydict in energyinputs:
        for stage in energydict:

            if stage not in combined:
                combined[stage] = {}
            if stage not in ("Total", "Total_ES"):
                for energytype in energydict[stage].keys():
                    if energytype == es:
                        total_es += energydict[stage][energytype]

                    if energytype not in combined[stage]:
                        combined[stage][energytype] = energydict[stage][energytype]
                    else:
                        combined[stage][energytype] = combined[stage][energytype] + energydict[stage][energytype]
                total += energydict[stage]['Total']

    combined['Total'] = total
    combined['Total_ES'] = total_es
    # print('total',total)
    return combined


def combine_energydicts_use(supplychainenergy, conversionenergy, _yield, es):
    combined = {}
    total = 0
    total_es = 0
    tempdict2 = {}
    for stage in supplychainenergy:
        if stage not in ("Total", 'Total_ES'):
            tempdict2[stage] = {}
            for energytype in supplychainenergy[stage]:
                tempdict2[stage][energytype] = supplychainenergy[stage][energytype] / _yield
        else:
            tempdict2[stage] = supplychainenergy[stage] / _yield

    energyinputs = [tempdict2, conversionenergy]

    if type(energyinputs) == dict:
        temp = energyinputs
        energyinputs = []
        for key, value in temp.items():
            energyinputs.append(value)

    for energydict in energyinputs:
        for stage in energydict:

            if stage not in combined:
                combined[stage] = {}
            if stage not in ("Total", "Total_ES"):
                for energytype in energydict[stage].keys():
                    if energytype == es:
                        total_es += energydict[stage][energytype]

                    if energytype not in combined[stage]:
                        combined[stage][energytype] = energydict[stage][energytype]
                    else:
                        combined[stage][energytype] = combined[stage][energytype] + energydict[stage][energytype]
                total += energydict[stage]['Total']

    combined['Total'] = total
    combined['Total_ES'] = total_es
    # print('total',total)
    return combined


def combine_emissionsdicts(emissionsdicts):
    # print(type(emissionsdicts))
    combined = {}
    total = 0
    for emissionsdict in emissionsdicts.values():
        # print(emissionsdict)
        for stage in emissionsdict:
            # print(emissionsdict[stage])
            if stage not in combined and stage != "Total":
                combined[stage] = emissionsdict[stage]
                total += emissionsdict[stage]
            elif stage != "Total":
                combined[stage] = emissionsdict[stage] + combined[stage]
                total += emissionsdict[stage]
    combined['Total'] = total
    # print('total',total)
    return combined


def combine_emissionsdicts_use(emissionsdicts, _yield):
    dictwithyield = {}
    for stage in emissionsdicts[1]:
        dictwithyield[stage] = emissionsdicts[1][stage] / _yield

    emissionsdicts[1] = dictwithyield
    # print(type(emissionsdicts))
    combined = {}
    total = 0
    for emissionsdict in emissionsdicts.values():
        # print(emissionsdict)
        for stage in emissionsdict:
            # print(emissionsdict[stage])
            if stage not in combined and stage != "Total":
                combined[stage] = emissionsdict[stage]
                total += emissionsdict[stage]
            elif stage != "Total":
                combined[stage] = emissionsdict[stage] + combined[stage]
                total += emissionsdict[stage]
    combined['Total'] = total
    # print('total',total)
    return combined


def assign_energysources_class(energyclass, energy_source, country):
    results = []
    baseemissions_E = {'Global': 'GlobalE'}
    baseemissions_NG = {'Global': 'GlobalNG'}
    for dict_bytype in [energyclass.embodied, energyclass.process, energyclass.operational]:
        dict = {}
        for energytype in dict_bytype:
            if energytype in ('Base_E') and energy_source.energytype == "E":
                dict[energy_source] = dict_bytype[energytype]
            elif energytype in ('Base_NG') and energy_source.energytype == "NG":
                dict[energy_source] = dict_bytype[energytype]
            elif energytype in ("Undefined_E", 'Base_E') and country in baseemissions_E:
                type = baseemissions_E[country]
                dict[type] = dict_bytype[energytype]
            elif energytype in ("Undefined_E", 'Base_E'):
                type = baseemissions_E['Global']
                dict[type] = dict_bytype[energytype]
            elif energytype in ("Undefined_NG", 'Base_NG') and country in baseemissions_NG:
                type = baseemissions_NG[country]
                dict[type] = dict_bytype[energytype]
            elif energytype in ("Undefined_NG", 'Base_NG'):
                type = baseemissions_NG['Global']
                dict[type] = dict_bytype[energytype]
            else:
                dict[energytype] = dict_bytype[energytype]
        results.append(dict)

    b = energy(results[0], results[1], results[2])
    return b


def final_use_impacts(result, uses):
    use = result.use

    if use in uses:
        vector_use = uses[use].vector

        if vector_use != result.vectors[-1]:
            conversion = result.vectors[-1] + '-' + vector_use
            Euo = conversions.instances[conversion].variables.energy_fixed
            Guo = conversions.instances[conversion].variables.emissions_fixed
            nuo = conversions.instances[conversion].variables._yield
            n = 1

            # calculate the totals
            Euo = assign_energysources_class(Euo, energysources.instances['GlobalE'], 'UK').calc_energy(
                conversions.instances[conversion], 1, result.energysource, 1)
            Guo = Guo.calc_emissions(energysources.instances, Euo, conversions.instances[conversion], 1, 1)
            Et = combine_energydicts_use(result.energy_total, Euo, nuo, result.energysource)
            Gt = combine_emissionsdicts_use({1: result.emissions_total, 2: Guo}, nuo)

        else:
            Euo = 0
            Guo = 0
            nuo = 1
            Et = result.energy_total
            Gt = result.emissions_total

        A = uses[use].input
        Eo = uses[use].energy
        Go = uses[use].emissions
        if np.isnan(Go):
            Go = 0
        Hu = uses[use].H2emissions
        if np.isnan(Hu):
            Hu = 0

        # if 'LH2' or 'NH3' in result.vectors:
        energy = {'H2': A * Et['Total'], 'Other': Eo}
        energy['H2 Process'] = A * Et['Process']['Total']
        energy['Total'] = energy['H2'] + energy['Other']
        energy_es_tot = Et['Total_ES'] * A

        emissions = {'H2': A * Gt['Total'], 'Other': Go, 'H2 emissions (process)': Hu * 12}
        if emissions['H2'] < 0 and vector_use != 'CH3OH':
            print(vector_use, A, Gt)
        if energy['Total'] < 0:
            print(vector_use, Et)
            exit()

        emissions['Total'] = emissions['H2'] + emissions['Other'] + emissions['H2 emissions (process)']
        # print(emissions['Total'])

    else:
        energy = {'Total': 'NA', 'H2': 'NA', 'Other': 'NA', 'H2 Process': 'NA'}
        energy_es_tot = {'Total': 'NA', 'H2': 'NA', 'Other': 'NA'}
        emissions = {'Total': 'NA', 'H2': 'NA', 'H2 emissions (process)': 'NA', 'Other': 'NA'}

    return {'Energy': energy, 'Emissions': emissions, 'Energy ES Total': energy_es_tot,
            'Energy H2 Supply': energy['H2 Process']}
