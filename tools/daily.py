import os
import numpy as np
from   glob import glob
from   astropy.table import Table
from   astropy.io.misc import hdf5 as Table5
import pandas as pd
import desispec.io
import psutil
import time
import pylab as plt

from tqdm import tqdm

import umap
from dadapy import Data
from pyfof import friends_of_friends as fof

plt.style.use('./tools/outliers.mplstyle')

class find_outliers():
    def __init__(self, data_params, umap_params, fof_params):
        self.release = data_params[0]
        self.survey  = data_params[1]
        self.night   = data_params[2]
        self.outpath = data_params[3]
        
        self.nn      = umap_params[0]
        self.md      = umap_params[1]
        self.me      = umap_params[2]
        
        self.ll      = fof_params[0]
        self.nm      = fof_params[1]
        
        self.outpath = f'{self.outpath}/{self.night}/{self.release}/{self.survey}'
        
        self.spec_file_b =   f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_b_spectra.csv'
        self.spec_file_r =   f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_r_spectra.csv'
        self.spec_file_z =   f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_z_spectra.csv'
        self.spec_file_brz = f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_brz_spectra.csv'
        
        self.wave_file_b =   f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_b_wave.csv'
        self.wave_file_r =   f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_r_wave.csv'
        self.wave_file_z =   f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_z_wave.csv'
        self.wave_file_brz = f'{self.outpath}/spectra/{self.night}_{self.release}_{self.survey}_brz_wave.csv'
        
        self.umap_file_b =   f'{self.outpath}/umap/{self.night}_{self.release}_{self.survey}_b_spectra_nn{self.nn}_md{self.md}_{self.me}.csv'
        self.umap_file_r =   f'{self.outpath}/umap/{self.night}_{self.release}_{self.survey}_r_spectra_nn{self.nn}_md{self.md}_{self.me}.csv'
        self.umap_file_z =   f'{self.outpath}/umap/{self.night}_{self.release}_{self.survey}_z_spectra_nn{self.nn}_md{self.md}_{self.me}.csv'
        self.umap_file_brz = f'{self.outpath}/umap/{self.night}_{self.release}_{self.survey}_brz_spectra_nn{self.nn}_md{self.md}_{self.me}.csv'
        
    def create_dataset(self, release, survey, night, globoutpath):
        new_releases = ['denali','everest','fuji','guadalupe','iron','daily']
        notrr_releases = ['andes','blanc','cascades','denali']
        
        tiles_path   = f'/global/cfs/cdirs/desi/spectro/redux/{release}/tiles'
        release_path = f'/global/cfs/cdirs/desi/spectro/redux/{release}'
        
        if release in new_releases:
            path = f'{tiles_path}/cumulative'
            
        exposures_file = f'{release_path}/exposures-{release}.csv'
        
        data = pd.read_csv(exposures_file)

        if 'FAPRGRM' in data.keys():
            data_tmp = data[data['FAPRGRM']!='backup']
        else:
            data_tmp = data.copy()

        data_tmp = data_tmp[data_tmp['SURVEY']==survey]
        tiles    = np.unique(data_tmp['TILEID'][data_tmp['NIGHT']==night])

        outpath = f'./{globoutpath}/spectra/'

        #################DANGER-> Just To_Test
        #tiles = tiles[0:1]
        #################DANGER

        if len(tiles)>0:
            os.makedirs(outpath, exist_ok=True)
        else:
            print('Not tiles found!')
            return 0

        scolumns = ['TARGETID','TILEID','PETAL_LOC','EXPID','FIBER','MJD','TARGET_RA','TARGET_DEC','FIBERSTATUS',
                    'FLUX_B','FLUX_R','FLUX_Z','FLUX_BRZ','SUMFLUX_B','SUMFLUX_R','SUMFLUX_Z','SUMFLUX_BRZ']
        zcolumns = ['targetid','z','zerr','zwarn','spectype','chi2','deltachi2']

        df_spec     = pd.DataFrame()
        df_wave_b   = pd.DataFrame()
        df_wave_r   = pd.DataFrame()
        df_wave_z   = pd.DataFrame()
        df_wave_brz = pd.DataFrame()

        if (not(os.path.isfile(self.spec_file_b)) or not(os.path.isfile(self.wave_file_b)) or 
            not(os.path.isfile(self.spec_file_r)) or not(os.path.isfile(self.wave_file_r)) or
            not(os.path.isfile(self.spec_file_z)) or not(os.path.isfile(self.wave_file_z)) or 
            not(os.path.isfile(self.spec_file_brz)) or not(os.path.isfile(self.wave_file_brz)) ):

            for tile in tiles:

                #print(f'Finding files on {path}/{tile}/{night}/')
                sfiles = np.sort(glob(f'{path}/{tile}/{night}/spectra*.fits*'))

                for sfile in sfiles:

                    #try:
                    sp    = desispec.io.read_spectra(sfile)
                    sp_pd = sp.fibermap.to_pandas()

                    sp_pd['FLUX_B'] = [i for i in sp.flux['b']]
                    sp_pd['FLUX_R'] = [i for i in sp.flux['r']]
                    sp_pd['FLUX_Z'] = [i for i in sp.flux['z']]
                    sp_pd['FLUX_BRZ'] = [np.concatenate([x,y,z])  for x,y,z in zip(sp.flux['b'],sp.flux['r'],sp.flux['z'])]

                    sum_flux_b = [sum(i) for i in sp_pd['FLUX_B']]
                    sum_flux_r = [sum(i) for i in sp_pd['FLUX_R']]
                    sum_flux_z = [sum(i) for i in sp_pd['FLUX_Z']]
                    sum_flux_brz = [sum(i) for i in sp_pd['FLUX_BRZ']]

                    sp_pd['SUMFLUX_B'] = sum_flux_b
                    sp_pd['SUMFLUX_R'] = sum_flux_r
                    sp_pd['SUMFLUX_Z'] = sum_flux_z
                    sp_pd['SUMFLUX_BRZ'] = sum_flux_brz

                    if release in notrr_releases:
                        zfile = sfile.replace('spectra', 'zbest')
                    else:
                        zfile = sfile.replace('spectra', 'redrock')
                        zfile = zfile.replace('.gz', '')
                        zfile = zfile.replace('.fits', '.h5')

                    zs = Table5.read_table_hdf5(zfile, path=None)

                    s_temp = sp_pd[scolumns]
                    z_temp = zs[zcolumns].to_pandas()
                    z_temp.rename(columns = {'targetid':'TARGETID', 'z':'Z', 'zerr':'ZERR', 'zwarn':'ZWARN', 'spectype':'SPECTYPE', 'chi2':'CHI2', 'deltachi2':'DELTACHI2'}, inplace = True)

                    df_spec = pd.concat([df_spec, pd.merge(s_temp, z_temp)], ignore_index=True)
                    print(df_spec)
                    #except:
                      #  print(f'Error reading {sfile}')

            if len(df_spec) > 0:
                df_spec = df_spec[~df_spec.astype(str).duplicated()]

                df_spec = df_spec[df_spec['FIBERSTATUS'] == 0]
                df_spec = df_spec.drop(['FIBERSTATUS'], axis=1)

                df_spec_b = df_spec[df_spec['SUMFLUX_B'] != 0]
                df_spec_b = df_spec_b.drop(['SUMFLUX_R','SUMFLUX_Z','SUMFLUX_BRZ','FLUX_R','FLUX_Z','FLUX_BRZ'], axis=1)
                df_spec_b = df_spec_b.rename(columns = {'FLUX_B':'FLUX'})
                df_spec_b.to_pickle(self.spec_file_b)
                del df_spec_b

                df_spec_r = df_spec[df_spec['SUMFLUX_R'] != 0]
                df_spec_r = df_spec_r.drop(['SUMFLUX_B','SUMFLUX_Z','SUMFLUX_BRZ','FLUX_B','FLUX_Z','FLUX_BRZ'], axis=1)
                df_spec_r = df_spec_r.rename(columns = {'FLUX_R':'FLUX'})
                df_spec_r.to_pickle(self.spec_file_r)
                del df_spec_r

                df_spec_z = df_spec[df_spec['SUMFLUX_Z'] != 0]
                df_spec_z = df_spec_z.drop(['SUMFLUX_B','SUMFLUX_R','SUMFLUX_BRZ','FLUX_B','FLUX_R','FLUX_BRZ'], axis=1)
                df_spec_z = df_spec_z.rename(columns = {'FLUX_Z':'FLUX'})
                df_spec_z.to_pickle(self.spec_file_z)
                del df_spec_z

                df_spec_brz = df_spec[df_spec['SUMFLUX_BRZ'] != 0]
                df_spec_brz = df_spec_brz.drop(['SUMFLUX_B','SUMFLUX_R','SUMFLUX_Z','FLUX_B','FLUX_R','FLUX_Z'], axis=1)
                df_spec_brz = df_spec_brz.rename(columns = {'FLUX_BRZ':'FLUX'})
                df_spec_brz.to_pickle(self.spec_file_brz)
                del df_spec_brz

                try:
                    df_wave_b['WAVE'] = sp.wave['b']
                    df_wave_r['WAVE'] = sp.wave['r']
                    df_wave_z['WAVE'] = sp.wave['z']
                    df_wave_brz['WAVE'] = np.concatenate([sp.wave['b'], sp.wave['r'], sp.wave['z']])
                except:
                    print('Wave error')
                    None

                df_wave_b.to_pickle(self.wave_file_b)
                df_wave_r.to_pickle(self.wave_file_r)
                df_wave_z.to_pickle(self.wave_file_z)
                df_wave_brz.to_pickle(self.wave_file_brz)

                del sp, sp_pd, zs, s_temp, z_temp, df_spec, df_wave_b, df_wave_r, df_wave_z, df_wave_brz
        else:
            print('Data set already exist!')


    def compute_umap(self, inpath, file, nn, md, me, reducer): 
        night    = file.split('/')[2]
        release  = file.split('/')[3]
        survey   = file.split('/')[4]
        label    = file.split('/')[-1][:-4]
        band     = label.split('_')[-2]
               
        # try:
        umapfile = f'{inpath}/umap/{label}_nn{nn}_md{md}_{me}.csv'
        figfile  = f'{inpath}/umap/{label}_nn{nn}_md{md}_{me}.png'
        
        if ((os.path.isfile(umapfile) == False) or (os.path.isfile(figfile) == False)):
            to = time.time()
            FLUX = list(pd.read_pickle(file)['FLUX'])
            #print(f'Time used reading flux={time.time()-to}')
            size = len(FLUX)

            # print('Computing Embedding\n')
            to = time.time()
            reducer.fit(FLUX)
            #print(f'Time fitting={time.time()-to}')

            to = time.time()
            embedding  = reducer.transform(FLUX)
            #print(f'Time embedding={time.time()-to}')

            memory     = psutil.Process().memory_info().rss / (1024 * 1024) /1000.0
            del reducer, FLUX      
            
            to = time.time()
            df_UMAP = pd.DataFrame()
            df_UMAP['X_UMAP'] = embedding[:,0]
            df_UMAP['Y_UMAP'] = embedding[:,1] 
            df_UMAP.to_pickle(umapfile)
            
            del df_UMAP

            SPECTYPE = np.array(pd.read_pickle(file)['SPECTYPE'], dtype='U13')
            classes = np.sort(np.unique(SPECTYPE))
            SPECTYPE[SPECTYPE == 'GALAXY'] = 0  # GALAXY=0  -   QSO=1    -  STAR=2
            SPECTYPE[SPECTYPE == 'QSO'] = 1
            SPECTYPE[SPECTYPE == 'STAR'] = 2
            SPECTYPE = np.array(SPECTYPE, dtype=int)

            colors = np.zeros(len(SPECTYPE)).astype(str)
            colors[SPECTYPE==0] = "#75bbfd"
            colors[SPECTYPE==1] = "#c20078"
            colors[SPECTYPE==2] = "#96f97b"

            fig = plt.figure(figsize=(20,6))
            xmin = min(embedding[:,0])-1
            xmax = max(embedding[:,0])+1
            ymin = min(embedding[:,1])-1
            ymax = max(embedding[:,1])+1

            size = 2.5

            plt.subplot(141)
            plt.title("All",size=20)
            plt.scatter(embedding[:,0], embedding[:,1], c=colors, cmap='Paired', s=size)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(alpha=0.5)

            plt.subplot(142)
            plt.title("Galaxy",size=20)
            plt.scatter(embedding[:,0][SPECTYPE==0], embedding[:,1][SPECTYPE==0], color="#75bbfd", cmap='Paired', s=size)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)  
            plt.xlabel('X')                         
            plt.grid(alpha=0.5)

            plt.subplot(143)
            plt.title("QSO",size=20)
            plt.scatter(embedding[:,0][SPECTYPE==1], embedding[:,1][SPECTYPE==1], color="#c20078", cmap='Paired', s=size)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax) 
            plt.xlabel('X')                           
            plt.grid(alpha=0.5)

            plt.subplot(144)
            plt.title("Star",size=20)
            plt.scatter(embedding[:,0][SPECTYPE==2], embedding[:,1][SPECTYPE==2], color="#96f97b" , cmap='Paired', s=size)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
            plt.xlabel('X')                        
            plt.grid(alpha=0.5)

            fig.suptitle(r'$N_n$='+str(nn)+r', $M_d$='+str(round(md,2))+r', $M_e$='+str(me.title())+ ", Band:"+band.upper(), size=25, y=1)
            plt.tight_layout()

            plt.savefig(f'{figfile}', bbox_inches='tight')
            #plt.savefig(f'{figfile}.pdf', bbox_inches='tight', dpi=450)
            plt.close()

            del fig, SPECTYPE, colors, embedding, xmin, xmax, ymin, ymax
            # except:
            #     None
            #print(f'Time plotting={time.time()-to}')

            return memory, size 
        
        else:
            print('Reduction already exist!')
            return 0, 0

    def ifof(self, inpath, obs_params, reduc_params, fof_params, plot=False, save=False, all_island=False, log=True):
        night, release, survey, band = obs_params[0], obs_params[1], obs_params[2], obs_params[3]
        nn, md, me = reduc_params[0], reduc_params[1], reduc_params[2]
        ll, mm = fof_params[0], fof_params[1]
        
        imagefile = f'{night}_{release}_{survey}_{band}_nn{nn}_md{md}_{me}_ll{ll}_mm{mm}'
        imagepath = f'./{inpath}/outliers_den'
        
        os.makedirs(imagepath, exist_ok=True)
        
        df_umap = pd.read_pickle(f'./{inpath}/umap/{night}_{release}_{survey}_{band}_spectra_nn{nn}_md{md}_{me}.csv')
        df_spec = pd.read_pickle(f'./{inpath}/spectra/{night}_{release}_{survey}_{band}_spectra.csv')
        df_wave = pd.read_pickle(f'./{inpath}/spectra/{night}_{release}_{survey}_{band}_wave.csv')

        XUMAP = np.array(df_umap['X_UMAP'])
        YUMAP = np.array(df_umap['Y_UMAP'])
        data = np.array([XUMAP,YUMAP], dtype=float).T

        aobs = 1.0
        afit = 0.5

        Z = np.array(df_spec['Z'])

        SPECTYPE = np.array(df_spec['SPECTYPE'], dtype='U13')
        classes = np.sort(np.unique(SPECTYPE))
        SPECTYPE[SPECTYPE == 'GALAXY'] = 0  # GALAXY=0  -   QSO=1    -  STAR=2
        SPECTYPE[SPECTYPE == 'QSO'] = 1
        SPECTYPE[SPECTYPE == 'STAR'] = 2
        SPECTYPE = np.array(SPECTYPE, dtype=int)

        colors = np.zeros(len(SPECTYPE)).astype(str)
        colors[SPECTYPE==0] = "#75bbfd"
        colors[SPECTYPE==1] = "#c20078"
        colors[SPECTYPE==2] = "#96f97b"

        FLUX = np.array(df_spec['FLUX'])
        size = len(FLUX)
        len_flux = len(FLUX[0])

        WAVE = np.array(df_wave['WAVE'])
        
        #-------Compute density
        data_den = Data(data, verbose=False)
        data_den.compute_density_kstarNN()
        density = data_den.log_den

        ## FOF algorithm to find the islands
        friends = fof(data, ll)
        # Apply a filter on the FOF:    friends>number_friends &  friends<data/# to remove big clusters
        friends_min = []
        spec_friends = []
        z_friends = []
        n_members = []  
        for f in friends:
            if (len(f)>=mm) & (len(f)<len(data)*0.2) & (np.average(density[f])<-6.5):
                friends_min.append(f)
                spec_friends.append(SPECTYPE[f])
                z_friends.append(Z[f])
                n_members.append(len(f))
                
        if len(n_members)!=0:
            mean = np.mean(n_members)
            std = np.std(n_members)
        else:
            mean = 0
            std = 0
           
        if log == True:
            print('{0} islands for ll {1:.2} with min {2} members!'.format(len(friends_min),ll,mm))
            print('{} outliers identified!'.format(sum(n_members)))                     
        if save == False:
            return friends_min, spec_friends, z_friends, len(friends_min), size, len_flux, n_members, sum(n_members), sum(n_members)*100/size, mean, std


        if all_island == True:
            if log == True:
                pbar =  tqdm(total=len(friends_min), desc=f'{night}/{release}/{survey}->{band}')

            fig = plt.figure(figsize=(8,8))

            for k, c in enumerate(classes):
                plt.scatter(XUMAP[SPECTYPE==k], YUMAP[SPECTYPE==k], c=colors[SPECTYPE==k], cmap='Paired', s=1, label=c, alpha=0.5)
            plt.legend(markerscale=8, fontsize=15)        
            plt.grid(alpha=0.5)
            plt.title('Night={}, Release={}, Survey={} \n Band={}, $L_l$={:.2f}, Islands={}, Outliers={}'.format(night, release.title(), survey.title(), band.upper(),ll,len(friends_min), sum(n_members)),size=20)

            for i in range(len(friends_min)):
                plt.scatter(XUMAP[friends_min[i]],YUMAP[friends_min[i]], s=5, c='black')
                if log == True:
                    pbar.update()
            if log == True:                
                pbar.close()

            xmin = min(XUMAP)-1
            xmax = max(XUMAP)+1
            ymin = min(YUMAP)-1
            ymax = max(YUMAP)+1        
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
            plt.xlabel('X')
            plt.ylabel('Y')
            if save==True:
                os.makedirs(f'{imagepath}',exist_ok=True)
                plt.savefig('{}/{}.png'.format(imagepath,imagefile), bbox_inches='tight')
                #plt.savefig('{}/{}.pdf'.format(imagepath,imagefile), bbox_inches='tight')
            if plot == True:
                plt.show()
            plt.close()
            del fig

        else:
            if log == True:
                pbar =  tqdm(total=len(friends_min), desc=f'{night}/{release}/{survey}->{band}')
            for i in range(len(friends_min)):
                fig = plt.figure(figsize=(35,9))
                gs = fig.add_gridspec(3,13)
                ax1 = fig.add_subplot(gs[0:,:4])

                for k, c in enumerate(classes):
                    plt.scatter(XUMAP[SPECTYPE==k], YUMAP[SPECTYPE==k], c=colors[SPECTYPE==k], cmap='Paired', s=1, label=c, alpha=0.5)

                plt.legend(markerscale=8, fontsize=15)        
                plt.grid(alpha=0.5)
                plt.title('Night={}, Release={}, Survey={} \n Band={}, L$_l$={:.2f}, Island={}/{}, Outliers={}'.format(night, release.title(), survey.title(), band.upper(),ll,i+1,len(friends_min),len(friends_min[i])),size=20)
                plt.scatter(XUMAP[friends_min[i]],YUMAP[friends_min[i]], s=5, c='black')
                xmin = min(XUMAP)-1
                xmax = max(XUMAP)+1
                ymin = min(YUMAP)-1
                ymax = max(YUMAP)+1        
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                plt.xlabel('X')
                plt.ylabel('Y')            

                for k, ii in enumerate(np.random.choice(friends_min[i],3,replace=False)):
                    ax2 = fig.add_subplot(gs[k:k+1,5:])

                    plt.plot(WAVE,FLUX[ii], alpha=aobs)

                    plt.title(str(np.array(df_spec['SPECTYPE'], dtype='U13')[ii])
                              +", "+"TILEID="+str(np.array(df_spec['TILEID'])[ii])
                              +", "+"PETAL="+str(np.array(df_spec['PETAL_LOC'])[ii])
                              +", "+"TARGETID="+str(int(np.array(df_spec['TARGETID'])[ii]))
                              +", "+"EXPID="+str(int(np.array(df_spec['EXPID'])[ii]))
                              +", "+"FIBER="+str(int(np.array(df_spec['FIBER'])[ii]))
                              +", "+"Z="+str(round(np.array(df_spec['Z'])[ii],3))
                              , fontsize=17)
                    # plt.legend(fontsize=12)
                    plt.grid(alpha=0.5)


                    if (k == 0) or (k == 1):
                        plt.setp(ax2.get_xaxis(), visible=False)
                        plt.setp(ax2.get_xticklabels(), visible=False)
                    if (k==1):
                        plt.ylabel("Flux $10^{-17}[erg/s\,cm^2\,\AA]$")

                plt.xlabel('$\lambda$ $[\AA]$')
                if save:
                    os.makedirs(f'{imagepath}',exist_ok=True)
                    plt.savefig('{}/{}_both_{}.png'.format(imagepath,imagefile,i+1), bbox_inches='tight')
                    #plt.savefig('{}/{}_both_{}.pdf'.format(imagepath,imagefile,i+1), bbox_inches='tight')
                if plot == True:
                    plt.show()
                plt.close()
                del fig

                fig, ax = plt.subplots(4,1, figsize=(15,20), sharey=True, sharex=True)
                for k, ii in enumerate(np.random.choice(friends_min[i],4,replace=False)):
                    plt.subplot(4,1,k+1)

                    plt.plot(WAVE,FLUX[ii], alpha=aobs)

                    plt.title(str(np.array(df_spec['SPECTYPE'])[ii])
                              +", "+"TILEID="+str(np.array(df_spec['TILEID'])[ii])
                              +", "+"PETAL="+str(np.array(df_spec['PETAL_LOC'])[ii])
                              +", "+"TARGETID="+str(int(np.array(df_spec['TARGETID'])[ii]))
                              +", "+"EXPID="+str(np.array(df_spec['EXPID'])[ii])
                              +", "+"FIBER="+str(int(np.array(df_spec['FIBER'])[ii]))
                              +", "+"Z="+str(round(np.array(df_spec['Z'])[ii],3))
                              , fontsize=14)
                    # plt.legend(fontsize=12)
                    plt.grid(alpha=0.5)

                    plt.ylabel("Flux $10^{-17}[erg/s\,cm^2\,\AA]$", fontsize=18)

                fig.suptitle('Band={}, $L_l$={:.2f}, Island={}/{}, Outliers={}'.format(band.upper(),ll,i+1,len(friends_min),len(friends_min[i])),y=0.92,size=20)

                plt.xlabel('$\lambda$ $[\AA]$', fontsize=18)
                if save:
                    plt.savefig('{}/{}_spec_{}.png'.format(imagepath,imagefile,i+1), bbox_inches='tight')
                    #plt.savefig('{}/{}_spec_{}.pdf'.format(imagepath,imagefile,i+1), bbox_inches='tight')
                # plt.show()
                plt.close()
                del fig

                fig = plt.figure(figsize=(8,8))
                for k, c in enumerate(classes):
                    plt.scatter(XUMAP[SPECTYPE==k], YUMAP[SPECTYPE==k], c=colors[SPECTYPE==k], cmap='Paired', s=3, label=c, alpha=0.5)

                plt.legend(markerscale=8, fontsize=15)        
                plt.grid(alpha=0.5)
                plt.title('Night={}, Release={}, Survey={} \n Band={}, L$_l$={:.2f}, Island={}/{},  Members={}'.format(night, release.title(), survey.title(), band.upper(),ll,i+1,len(friends_min),len(friends_min[i])),size=20)
                plt.scatter(XUMAP[friends_min[i]],YUMAP[friends_min[i]], s=10, c='black')
                xmin = min(XUMAP)-1
                xmax = max(XUMAP)+1
                ymin = min(YUMAP)-1
                ymax = max(YUMAP)+1        
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.tight_layout()
                if save:
                    plt.savefig('{}/{}_umap_{}.png'.format(imagepath,imagefile,i+1), bbox_inches='tight')
                    #plt.savefig('{}/{}_umap_{}.pdf'.format(imagepath,imagefile,i+1), bbox_inches='tight')
    #             plt.show()
                plt.close()
                del fig
                if log == True:
                    pbar.update()
            if log == True:
                pbar.close()

        del data, df_umap, df_spec, df_wave, WAVE, FLUX, SPECTYPE, Z
        #this function return,
        
        return friends_min, spec_friends, z_friends, len(friends_min), size, len_flux, n_members, sum(n_members), sum(n_members)*100/size, mean, std

       
    def run(self, save=True):  
        
        print('#######################')
        print('Creating dataset')
        print('#######################')
        r = self.create_dataset(self.release, self.survey, self.night, self.outpath)
        if r==0:
            print('This DR does not exist!')
            return 0
        print(end='\n\n')

        os.makedirs(f'{self.outpath}/umap', exist_ok=True)
        print('#######################')
        print(f'Finding outliers')
        print(f'UMAP params Nn={self.nn}, Md={self.md}, Me={self.me}')
        print(f'FoF params  Ll={self.ll}, Nm={self.nm}')
        print('#######################')
        self.model = umap.UMAP(n_neighbors=self.nn, min_dist=self.md, metric=self.me, random_state=42, low_memory=False)

        total_memory = 0
        total_time   = 0

        start_time, t_time = 0, 0
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print('Band B ...')
        start_time = time.time()
        memory, size = self.compute_umap(self.outpath, self.spec_file_b, self.nn, self.md, self.me, self.model)
        self.ifof(self.outpath, (self.night,self.release,self.survey,'b'), (self.nn,self.md,self.me), (self.ll,self.nm), plot=False, save=True, all_island=False, log=True)    
        t_time = time.time() - start_time
        print(f'Time={t_time:.2f}s - Memory_used={memory:.2f}GB')
        total_time   += t_time
        total_memory += memory
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print(end='\n\n')      

        start_time, t_time = 0, 0
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print('Band R ...')
        start_time = time.time()
        memory, size = self.compute_umap(self.outpath, self.spec_file_r, self.nn, self.md, self.me, self.model)
        self.ifof(self.outpath, (self.night,self.release,self.survey,'r'), (self.nn,self.md,self.me), (self.ll,self.nm), plot=False, save=True, all_island=False, log=True)  
        t_time = time.time() - start_time
        print(f'Time={t_time:.2f}s - Memory_used={memory:.2f}GB')
        total_time   += t_time
        total_memory += memory    
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print(end='\n\n')

        start_time, t_time = 0, 0
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print('Band Z ...')
        start_time = time.time()
        memory, size = self.compute_umap(self.outpath, self.spec_file_z, self.nn, self.md, self.me, self.model)
        self.ifof(self.outpath, (self.night,self.release,self.survey,'z'), (self.nn,self.md,self.me), (self.ll,self.nm), plot=False, save=True, all_island=False, log=True)  
        t_time = time.time() - start_time
        print(f'Time={t_time:.2f}s - Memory_used={memory:.2f}GB')
        total_time   += t_time
        total_memory += memory    
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print(end='\n\n')

        start_time, t_time = 0, 0
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print('Band BRZ ...')
        start_time = time.time()
        memory, size = self.compute_umap(self.outpath, self.spec_file_brz, self.nn, self.md, self.me, self.model)
        self.ifof(self.outpath, (self.night,self.release,self.survey,'brz'), (self.nn,self.md,self.me), (self.ll,self.nm), plot=False, save=True, all_island=False, log=True)  
        t_time = time.time() - start_time
        print(f'Time={t_time:.2f}s - Memory_used={memory:.2f}GB')
        total_time   += t_time
        total_memory += memory    
        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print(end='\n\n')  

        print(f'Total Time={total_time:.2f}s - Total Memory_used={total_memory:.2f}GB - Number of spectra={size*4}')