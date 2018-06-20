"""
An example script on how to preprocess a trackingNtuple.root file to a flat format for machine learning
"""

import ROOT
import pandas as pd
import numpy as np
from ntuple import * 
from root_pandas import to_root

def main():
	ntuple = TrackingNtuple("trackingNtuple.root")

	#Variables of the track
	#trk_isTrue is gen level info, other variables are from the reconstruction
        track_vars_ = ['trk_isTrue','trk_mva','trk_pt','trk_eta','trk_dxy','trk_dz','trk_dxyClosestPV','trk_dzClosestPV',
        'trk_ptErr','trk_etaErr','trk_lambdaErr','trk_dxyErr','trk_dzErr','trk_nChi2','trk_ndof','trk_nLost','trk_lambda',
        'trk_nPixel','trk_nStrip','trk_nPixelLay','trk_nStripLay','trk_n3DLay','trk_nLostLay','trk_algo']

	#Variables of hits in the track
	#Be sure to append the hit variables below in this order as well if you make changes
	hit_vars_ = ['Chi2','x','y','z','residual_x','residual_y','pull_x','pull_y']


	#To have the inputs to a network be of constant length, one has to decide a number of hits to include for each track
	nHits = 15
	hit_vars_ = [y+"_"+str(x) for x in range(nHits) for y in hit_vars_]

	#Dataframe to store the flattened data
	df=pd.DataFrame(columns=track_vars_+hit_vars_)


	for event in ntuple:
		hitinfo=event.hitInfo()
		for track in event.tracks():
			event_row=[]

			event_row.append(track.isTrue())
			event_row.append(track.mva())
			event_row.append(track.pt())
                        event_row.append(track.eta())
			event_row.append(getattr(track, "lambda")())
                        event_row.append(track.dxy())
                        event_row.append(track.dz())
                        event_row.append(track.dxyClosestPV())
                        event_row.append(track.dzClosestPV())
                        event_row.append(track.ptErr())
                        event_row.append(track.etaErr())
                        event_row.append(track.lambdaErr())
                        event_row.append(track.dxyErr())
                        event_row.append(track.dzErr())
                        event_row.append(track.nChi2())
                        event_row.append(track.ndof())
			event_row.append(int(track.nLost()))
                        event_row.append(int(track.nPixel()))
                        event_row.append(int(track.nStrip()))
                        event_row.append(int(track.nPixelLay()))
                        event_row.append(int(track.nStripLay()))
                        event_row.append(int(track.n3DLay()))
                        event_row.append(int(track.nLostLay()))
                        event_row.append(int(track.algo()))

			#Padding if number of hits in track less than nHits
			hitIdx=track.hitInfo_hitIdx()
			for hit in range(nHits):
				if hit<len(hitIdx):
					event_row.append(hitinfo.Chi2()[hitIdx[hit]])
                                        event_row.append(hitinfo.x()[hitIdx[hit]])
                                        event_row.append(hitinfo.y()[hitIdx[hit]])
                                        event_row.append(hitinfo.z()[hitIdx[hit]])
					event_row.append(hitinfo.residual_x()[hitIdx[hit]])
                                        event_row.append(hitinfo.residual_y()[hitIdx[hit]])
                                        event_row.append(hitinfo.pull_x()[hitIdx[hit]])
                                        event_row.append(hitinfo.pull_y()[hitIdx[hit]])
				else:
					#Choose a padding number that is clearly outside the phase space of valid hits
					#Still this can mislead the ML method
                                        event_row.append(-99.)
                                        event_row.append(-99.)
                                        event_row.append(-99.)
                                        event_row.append(-99.)
                                        event_row.append(-99.)
                                        event_row.append(-99.)
                                        event_row.append(-99.)
                                        event_row.append(-99.)


			df=df.append(pd.Series(event_row,index=track_vars_+hit_vars_),ignore_index=True)

	#Saving dataframe. This adds an additional '__index__' branch that should be dropped when using data for training
	df.to_root('flattened_trackingNtuple.root','ttree')

if __name__ == "__main__":
    main()
