#!/usr/bin/python
import math
import re
import sys
###############Flywheel Control Class##################
class flywheel:
    on = True;
    w = 0 #angular velocity
    r = 0 #radius
    m = 0 #mass
    #Note: efficency assumes about 12.5W idle power consumption
    effIdle = .000000023 #%energy lost / sec
    effConv = .9 #% energy maintained in conversion
    energy = 0 #energy of a single flywheel
    energyH = 0 #energy lost to heat from efficency
    energyMax = 0 #Max energy of flywheel array
    energyMin = 0 #Min energy of flywheel array
    energyGM = 0 #energy pulled from the grid
    energyGP = 0 #energy given to grid
    energyL = 0 #energy lost to heat
    moment = 0 #moment of intertia of a single flywheel
    wheelnum = 0 #number of flywheels
    ###############Flywheel Constructor####################
    def __init__(self, w, r, m, wheelnum):
        self.moment = .5 * m * r * r
        self.energy = .5 * self.moment * w * w * wheelnum
        self.energyMax = .5 * self.moment * w * w * wheelnum
        self.w = w
        self.r = r
        self.m = m
        self.wheelnum = wheelnum

    ###############Update Function for Flywheel Energy#####
    def updateW(self):
        self.w = math.sqrt(2*self.energy/self.moment)

    ###############Update Function for Flywheel Array######
    def update(self, delT, inputS, demandS):
        #Energy conversion loss
        input  = inputS * self.effConv
        demand = demandS * (1 / self.effConv)

        self.energyH += inputS - input
        self.energyH += demand - demandS
        
        #Idle energy loss
        self.energyH += (self.effIdle)*self.energy*delT
        self.energy  -= (self.effIdle)*self.energy*delT

        #input is power input into flywheel array
        #demand is power output required from flywheel array
        #delT is time in seconds for step
        energyCh = 0 #change in energy of flywheel array
        energyCh = (input - demand)*delT

#        print energyCh
#        print self.on
            
        if(self.on):
            #if change in energy makes energy drop below minimum
            if(self.energy - self.energyMin + energyCh <= 0):
                self.energyGM -= self.energy - self.energyMin + energyCh
                self.on = False
                self.energy = self.energyMin
                self.updateW()
            #if change in energy makes energy rise above maximum
            elif((self.energy + energyCh) - self.energyMax >= 0):
                self.energyGP = (self.energy + energyCh) - self.energyMax
                self.energy = self.energyMax
                self.updateW()
            #if change in energy changes within normal range
            else:
                self.energy += energyCh
                self.updateW()                
        #if change in energy changes within normal range                    
        else:
            #if change in energy makes energy remain below minimum
            if(self.energy - self.energyMin + energyCh <= 0):
                self.energyGM -= self.energy - self.energyMin + energyCh
            #if change in energy makes energy rise above maximum                
            elif((self.energy + energyCh) - self.energyMax >= 0):
                self.energyGP = (self.energy + energyCh) - self.energyMax
                self.energy = self.energyMax
                self.on = True
                self.updateW()
            #if change in energy changes within normal range
            else:
                self.on = True
                self.energy += energyCh - self.energyMin
                if(self.energy <= 0):
                    print "OMG!  UNIVERSE BLOWS UP!"
                self.updateW()



###############Main Section############################
fly = flywheel(2530/2.0,4,30, 1)
#print fly.energyMax
#sys.exit()
#for i in range(0,6):
#    print fly.energy
#    fly.update(3600, 0, 3000) #3.2W to maintain
#for i in range(0,6):
#    print fly.energy
#    fly.update(3600, 7000, 15000) #3.2W to maintain
#for i in range(0,6):
#    print fly.energy
#    fly.update(3600, 7000, 22400) #3.2W to maintain
#for i in range(0,6):   
#    print fly.energy
#    fly.update(3600, 0, 3000) #3.2W to maintain

f = open('./Input_Demand_Data.txt', 'r')

inScale = 1
deScale = 1 #56.0/3.0

list = f.readlines()

for line in list:
    data = re.split(' ', line.rstrip('\r\n'))
#    print data
    if len(data) == 3:
        print fly.energy
        fly.update(int(data[0]), int(data[2]) * inScale, int(data[1]) * deScale)
for line in list:
    data = re.split(' ', line.rstrip('\r\n'))
#    print data
    if len(data) == 3:
        print fly.energy
        fly.update(int(data[0]), int(data[2]) * inScale, int(data[1]) * deScale)

totalcost = 0
i = 0
for line in list:
    data = re.split(' ', line.rstrip('\r\n'))
    value = int(data[2]) - int(data[1])
    if(i%24<6):
        if(value >=0):
            totalcost -= value *.0001 * 356 * 10 
        elif(value < 0):
            totalcost += value *.0001 * 356 * 10 
    elif(i%24<18):
        if(value >=0):
            totalcost -= value *.0002 * 356 * 10 
        elif(value < 0):
            totalcost += value *.0002 * 356 * 10 
    if(i%24<24):
        if(value >=0):
            totalcost -= value *.0001 * 356 * 10 
        elif(value < 0):
            totalcost += value *.0001 * 356 * 10 
    i += 1

print "Total Cost Solar   : " + str(-totalcost)
print "Total Cost Flywheel: " + str((fly.energyGP/(2 * 24 * 3600))/1000 * .2 * 365 * 20 * 24)

print "Energy Deficit: " + str(fly.energyGM)
print "Energy Surplus: " + str(fly.energyGP)
