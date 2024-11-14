import numpy as np

def get_PN_Std_Bhandawat(PNactivity):
    # for a numPNs x numOdors matrix, returns the estimated
    # standard deviation for each response, based on Bhandawat et
    # al 2007 Fig 1e.
    numPNs, numOdors = PNactivity.shape
    result = np.zeros_like(PNactivity)
    
    # column 1 is bin centers (mean spike rate in Hz)
    # column 2 is the average st.dev. for 50 ms windows with that
    # mean spike rate (in Hz)
    BhandawatData = np.array([
          [ 10, 2.929936306 ],
          [ 30, 6.904458599 ],
          [ 50, 8.687898089 ],
          [ 70, 10.31847134 ],
          [ 90, 11.2611465  ],
          [ 110, 11.69426752 ],
          [ 130, 11.69426752 ],
          [ 150, 10.70063694 ],
          [ 170, 9.78343949 ],
          [ 190, 9.732484076 ],
          [ 210, 8.866242038 ],
          [ 230, 8.687898089 ],
          [ 250, 7.363057325 ],
          [ 270, 8.152866242 ],
          [ 290, 10.67515924 ],
          [ 310, 9.910828025 ]
                        ])
    for i in range(numPNs):
        for j in range(numOdors):
            idx = np.argmin(np.abs(BhandawatData[:,0]-PNactivity[i,j]))
            result[i,j] = BhandawatData[idx,1]
    return result
    
    
def propagateORN2PN(ORNactivity=None)
    """get PN activity from ORNs. Accepts ORN input as argument, as a K-by-M 
     format with K response patterns in M ORNs (or glomeruli)
     When no argument is given, it will operate on the Hallem dataset"""
    if not ORNactivity: # read ORNs from Hallem dataset
        from pandas import read_csv
        hd = read_csv('hallem_and_carlson_2006.csv')
        hd = hd.set_index('odor')
        odorList = hd.index.to_list()
        spontFrate = hd.loc['spontaneous firing rate']
        ORNactivity = hd.drop(index=['spontaneous firing rate']).to_numpy()
    else:
        spontFrate = None
        odorList = []
    m = 10.63 #gain of lateral inhibition in AL
    Rmax = 165. #maximum PN response
    sigma = 12. #non-linearity parameter of ORN to PN response function
    s = m * np.sum(ORNactivity,axis=1)/190.
    s = s.reshape(-1,1)
    
    PNactivity = Rmax * ORNactivity**1.5 /( 
                    ORNactivity**1.5 +s**1.5 +sigma**1.5 )
    
    return (PNactivity, spontFrate, odorList)
    
    
