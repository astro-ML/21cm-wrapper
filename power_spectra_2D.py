import numpy as np
import matplotlib.pyplot as plt
import py21cmfast as p21c

'''
# read some test cubes
lightcone = p21c.outputs.LightCone.read("./Lightcones/LightCone_z4.0.h5")
#coeval8 = p21c.outputs.Coeval.read("./Lightcones/Coeval_z8.0.h5")
coeval10 = p21c.outputs.Coeval.read("./Lightcones/Coeval_z10.0.h5")
#coeval15 = p21c.outputs.Coeval.read("./Lightcones/Coeval_z15.0.h5")
#coeval20 = p21c.outputs.Coeval.read("./Lightcones/Coeval_z20.0.h5")

#lc_1 = './Lightcones/run895.npz' # BOX_LEN = 200 , DIM = 140
#lc_2 = './Lightcones/run996.npz'
lc_3 = './Lightcones/run795.npz'
#lc_4 = './Lightcones/run695.npz'
#lc_5 = './Lightcones/run395.npz'
#lc_6 = './Lightcones/run1735.npz'

gxH_redshifts = np.load('./Lightcones/redshifts5.npy')
'''

# reading the LCs and computing the redshifts
def read_image(files):
    cone = np.load(files)
    image = cone['image']
    return image

def read_gxH(files):
    cone = np.load(files)
    gxH = cone['gxH']
    return gxH

def compute_redshifts(files,return_Ts=True):
    cone = np.load(files)
    image = cone['image']
    label = cone['label']
    
    cosmo_params = p21c.CosmoParams(OMm=label[1])
    astro_params = p21c.AstroParams(INHOMO_RECO=True)
    user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
    flag_options = p21c.FlagOptions()
    sim_lightcone=p21c.LightCone(5.,user_params,cosmo_params,astro_params,flag_options,0,{"brightness_temp":image},35.05)
    redshifts=sim_lightcone.lightcone_redshifts

    return redshifts


# computes the 2D PS given a Coeval object
def compute_2D_PS_Coeval(field):
    DIM = field.user_params.HII_DIM # simulation grid
    BOX_LEN = field.user_params.BOX_LEN 
    cell_size = BOX_LEN/DIM 
    print('resolution is:', cell_size)
    d_cell = cell_size/2/np.pi
    sample_rate = 1/d_cell
    DIM_MIDDLE = int(DIM/2+1)

    kbox = np.fft.rfftfreq(int(DIM), d=d_cell)
    #print('# of k bins: ', np.shape(kbox))

    # dim, binning etc
    k_factor = 2.0
    V= BOX_LEN**3 # 21cmFAST C code
    DELTA_K = kbox[1]
    #print('deltak',DELTA_K)
    k_first_bin_ceil = DELTA_K
    k_max=DELTA_K*DIM # kbox[-1] # kbox[-1] is max available, or DELTA_K*DIM
    #print('kmax',k_max)
    #print('kbox[-1]',kbox[-1]) 

    # logarithmic counting
    NUM_BINS = 0
    k_floor = 0
    k_ceil = k_first_bin_ceil
    while (k_ceil < k_max):
        NUM_BINS += 1
        k_floor=k_ceil
        k_ceil = k_ceil*k_factor
        
    print('NUM_BINS count', NUM_BINS)

    # initialise 2d
    in_bin_ct = [0 for i in range(NUM_BINS)]
    in_bin_ct_perp = [0 for i in range(NUM_BINS)]
    in_bin_ct_par = [0 for i in range(NUM_BINS)]
    p_box = np.zeros((NUM_BINS, NUM_BINS))
    p_noise = [0 for i in range(NUM_BINS)]
    k_ave = [0 for i in range(NUM_BINS)]
    k_perp_ave = [0 for i in range(NUM_BINS)]
    k_par_ave = [0 for i in range(NUM_BINS)]

    # scale field
    field_scaled = field.brightness_temp * cell_size**3
    # fourier transform of field
    deldel_field = np.fft.rfftn(field_scaled) 
    field_amplitudes = np.abs(deldel_field)**2 

    # loop to construct  2D Pk
    for n_x in range(DIM):
        if (n_x>DIM_MIDDLE):
            k_x =(n_x-DIM) * DELTA_K 
        else:
            k_x = n_x * DELTA_K
        for n_y in range(DIM):
            if (n_y>DIM_MIDDLE):   #if (n_x>DIM_MIDDLE):
                k_y =(n_y-DIM) * DELTA_K
            else:
                k_y = n_y * DELTA_K
            for n_z in range(DIM_MIDDLE):
                k_z = n_z * DELTA_K

                k_perp = np.sqrt(k_x*k_x + k_y*k_y)
                k_par = np.sqrt(k_z*k_z)
                k_mag = np.sqrt(k_perp*k_perp + k_par*k_par)

                k_floor = 0
                k_ceil = k_first_bin_ceil
                ct = 0
                ct_perp = 0
                ct_par = 0
                k_perp_floor = 0
                k_par_floor = 0
                k_perp_ceil = k_first_bin_ceil
                k_par_ceil = k_first_bin_ceil

                while k_ceil < k_max :
                    if ((k_mag >= k_floor) and (k_mag <= k_ceil)):
                        in_bin_ct[ct] += 1
                        while k_perp_ceil < k_max :
                            
                            if ((k_perp>=k_perp_floor) and (k_perp < k_perp_ceil)):
                                in_bin_ct_perp[ct_perp] += 1

                                while k_par_ceil < k_max :
                                    if ((k_par >= k_par_floor) and (k_par <= k_par_ceil)):
                                        in_bin_ct_par[ct_par] += 1
                                        p_box[ct_par][ct_perp] += np.array(pow(k_perp,2)*np.array(k_par)*field_amplitudes[n_x,n_y,n_z]/(2.0*np.pi**2*V))
                                        k_par_ave[ct_par] += k_par
                                        break
                                    ct_par+=1
                                    k_par_floor=k_par_ceil
                                    k_par_ceil*=k_factor  
                                
                                k_perp_ave[ct_perp] += k_perp
                                break

                            ct_perp+=1
                            k_perp_floor=k_perp_ceil
                            k_perp_ceil*=k_factor
                        
                        k_ave[ct] += k_mag
                        break

                    ct += 1
                    k_floor = k_ceil
                    k_ceil *= k_factor
                    

    k_par_ave = np.array(k_par_ave)
    k_perp_ave = np.array(k_perp_ave)
    in_bin_ct_par = np.array(in_bin_ct_par)
    in_bin_ct_perp = np.array(in_bin_ct_perp)


    # Initialize p_box_norm array
    p_box_norm = np.zeros_like(p_box)
    k_perp_norm = np.zeros_like(k_perp_ave)
    k_par_norm = np.zeros_like(k_par_ave)

    # Compute p_box_norm
    for ct_par in range(NUM_BINS):
        for ct_perp in range(NUM_BINS):
            if k_par_ave[ct_par] != 0 and k_perp_ave[ct_perp] != 0 and in_bin_ct_par[ct_par] != 0 and in_bin_ct_perp[ct_perp] != 0:
                p_box_norm[ct_par][ct_perp] = p_box[ct_par][ct_perp] / (in_bin_ct_perp[ct_perp] + in_bin_ct_par[ct_par])
                k_perp_norm[ct_perp] = k_perp_ave[ct_perp] / in_bin_ct_perp[ct_perp] 
                k_par_norm[ct_par] = k_par_ave[ct_par] / in_bin_ct_par[ct_par] 

    return k_perp_norm, k_par_norm, p_box_norm, field.redshift
    

# plots the 2D PS for coeval object
def plot_2D_PS_Coeval(coeval_PS):
    k_perp_norm = coeval_PS[0]
    k_par_norm = coeval_PS[1]
    p_box_norm = coeval_PS[2]
    z_mean = coeval_PS[3]
    # Create 2D plot (contour plot)
    K_PERP, K_PAR = np.meshgrid(k_perp_norm, k_par_norm)
    mesh = plt.pcolormesh(K_PERP, K_PAR, p_box_norm, cmap='BuPu', shading='auto', norm='log')
    colorbar = plt.colorbar(mesh, label=r'$\log_{10}(\Delta^2(k_\perp, k_\parallel))\left[\mathrm{mK^2}\right]$')
    #colorbar.set_ticklabels([0, 1, 2])

    # Label axes
    plt.xlabel(r'$k_\perp \left[\mathrm{Mpc}^{-1}\right]$')
    plt.ylabel(r'$k_\parallel \left[\mathrm{Mpc}^{-1}\right]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.05, 1.1)
    plt.ylim(0.05, 1.1)
    plt.minorticks_off()
    plt.title(r'$\bar z$'f'={z_mean}')
    plt.grid(True)
    plt.show()


# Now for Lightcones

# computing the 2D PS for LC chunks
def compute_2D_PS_chunk(field, BOX_LEN, DIM, chunklen):
    cell_size = BOX_LEN/DIM 
    print('resolution is:', np.round(cell_size, 2))
    d_cell = cell_size/2/np.pi
    sample_rate = 1/d_cell
    DIM_MIDDLE = int(DIM/2+1)

    kbox = np.fft.rfftfreq(int(DIM), d=d_cell)
    #print('# of k bins: ', np.shape(kbox))

    # dim, binning etc
    k_factor = 2.0 
    V= BOX_LEN**2 * chunklen # Volume in Mpc**3
    DELTA_K = kbox[1]
    #print('deltak',DELTA_K)
    k_first_bin_ceil = DELTA_K
    k_max=DELTA_K*DIM # kbox[-1] # kbox[-1] is max available, or DELTA_K*DIM
    #print('kmax',k_max)
    #print('kbox[-1]',kbox[-1]) 

    # logarithmic counting
    NUM_BINS = 0
    k_floor = 0
    k_ceil = k_first_bin_ceil
    while (k_ceil < k_max):
        NUM_BINS += 1
        k_floor=k_ceil
        k_ceil = k_ceil*k_factor
        
    print('NUM_BINS count', NUM_BINS)

    # initialise 2d
    in_bin_ct = [0 for i in range(NUM_BINS)]
    in_bin_ct_perp = [0 for i in range(NUM_BINS)]
    in_bin_ct_par = [0 for i in range(NUM_BINS)]
    p_box = np.zeros((NUM_BINS, NUM_BINS))
    p_noise = [0 for i in range(NUM_BINS)]
    k_ave = [0 for i in range(NUM_BINS)]
    k_perp_ave = [0 for i in range(NUM_BINS)]
    k_par_ave = [0 for i in range(NUM_BINS)]

    # scale field
    field_scaled = field * cell_size**3
    # fourier transform of field
    deldel_field = np.fft.rfftn(field_scaled) 
    field_amplitudes = np.abs(deldel_field)**2 

    # loop to construct  2D Pk
    for n_x in range(DIM):
        if (n_x>DIM_MIDDLE):
            k_x =(n_x-DIM) * DELTA_K 
        else:
            k_x = n_x * DELTA_K
        for n_y in range(DIM):
            if (n_y>DIM_MIDDLE):   #if (n_x>DIM_MIDDLE):
                k_y =(n_y-DIM) * DELTA_K
            else:
                k_y = n_y * DELTA_K
            for n_z in range(DIM_MIDDLE):
                k_z = n_z * DELTA_K

                k_perp = np.sqrt(k_x*k_x + k_y*k_y)
                k_par = np.sqrt(k_z*k_z)
                k_mag = np.sqrt(k_perp*k_perp + k_par*k_par)

                k_floor = 0
                k_ceil = k_first_bin_ceil
                ct = 0
                ct_perp = 0
                ct_par = 0
                k_perp_floor = 0
                k_par_floor = 0
                k_perp_ceil = k_first_bin_ceil
                k_par_ceil = k_first_bin_ceil

                while k_ceil < k_max :
                    if ((k_mag >= k_floor) and (k_mag <= k_ceil)):
                        in_bin_ct[ct] += 1
                        while k_perp_ceil < k_max :
                            
                            if ((k_perp>=k_perp_floor) and (k_perp < k_perp_ceil)):
                                in_bin_ct_perp[ct_perp] += 1

                                while k_par_ceil < k_max :
                                    if ((k_par >= k_par_floor) and (k_par <= k_par_ceil)):
                                        in_bin_ct_par[ct_par] += 1
                                        p_box[ct_par][ct_perp] += np.array(pow(k_perp,2)*np.array(k_par)*field_amplitudes[n_x,n_y,n_z]/(2.0*np.pi**2*V))
                                        k_par_ave[ct_par] += k_par
                                        break
                                    ct_par+=1
                                    k_par_floor=k_par_ceil
                                    k_par_ceil*=k_factor  
                                
                                k_perp_ave[ct_perp] += k_perp
                                break

                            ct_perp+=1
                            k_perp_floor=k_perp_ceil
                            k_perp_ceil*=k_factor
                        
                        k_ave[ct] += k_mag
                        break

                    ct += 1
                    k_floor = k_ceil
                    k_ceil *= k_factor
                    

    k_par_ave = np.array(k_par_ave)
    k_perp_ave = np.array(k_perp_ave)
    in_bin_ct_par = np.array(in_bin_ct_par)
    in_bin_ct_perp = np.array(in_bin_ct_perp)


    # Initialize p_box_norm array
    p_box_norm = np.zeros_like(p_box)
    k_perp_norm = np.zeros_like(k_perp_ave)
    k_par_norm = np.zeros_like(k_par_ave)

    # Compute p_box_norm
    for ct_par in range(NUM_BINS):
        for ct_perp in range(NUM_BINS):
            if k_par_ave[ct_par] != 0 and k_perp_ave[ct_perp] != 0 and in_bin_ct_par[ct_par] != 0 and in_bin_ct_perp[ct_perp] != 0:
                p_box_norm[ct_par][ct_perp] = p_box[ct_par][ct_perp] / (in_bin_ct_perp[ct_perp] + in_bin_ct_par[ct_par])
                k_perp_norm[ct_perp] = k_perp_ave[ct_perp] / in_bin_ct_perp[ct_perp] 
                k_par_norm[ct_par] = k_par_ave[ct_par] / in_bin_ct_par[ct_par] 

    return k_perp_norm, k_par_norm, p_box_norm


def compute_2D_PS_LC(field, nchunks=2): # here field is LC object
    data = []
    z_mean = []
    xHI_mean = []
    Tb_mean = []
    DIM = field.user_params.HII_DIM # simulation grid
    BOX_LEN = field.user_params.BOX_LEN # physical dimension
    chunk_indices = list(range(0,field.n_slices,round(field.n_slices / nchunks),)) # deviding data into chunks 
    
    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(field.n_slices) # when dividing the data we may lose the last entry so we append it manually
   
    for i in range(nchunks):
        print(f'Chunk #{i+1}')
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunklen = (end - start) * field.cell_size # the physical chuncklength in MPc
        # computing the PS
        k_perp, k_par, power = compute_2D_PS_chunk(field.brightness_temp[:, :, start:end], BOX_LEN=BOX_LEN, DIM=DIM, chunklen=chunklen)
        # append power and compute mean redshift, ionized fraction and brightness temp per chunk
        data.append({"k_perp": k_perp, "k_par": k_par, "delta": power})
        z_mean.append(np.mean(field.lightcone_redshifts[start:end]))
        print('Mean redshift:', np.round(z_mean[i], 2))
        xHI_mean.append(np.mean(field.xH_box[:,:,start:end]))
        Tb_mean.append(np.mean(field.brightness_temp[:,:,start:end]))
        print('-----------------------------')
        
    return data, z_mean, xHI_mean, Tb_mean


# plots the 2D PS for LC object
def plot_2D_PS_LC(lightcone_chunks):
    nchunks = len(lightcone_chunks[0])
    lightcone_PS = lightcone_chunks[0]
  
    lightcone_redshifts = lightcone_chunks[1]
    lightcone_ionized_frac = lightcone_chunks[2]
    lightcone_global_temp = lightcone_chunks[3]

    #fig, axs = plt.subplots(nchunks, figsize=(8, 12), constrained_layout=True)
    #plt.rcParams['text.usetex'] = True

    for i in range(nchunks):
        K_PAR, K_PERP = np.meshgrid(lightcone_PS[i]['k_perp'], lightcone_PS[i]['k_par'])
        mesh = plt.pcolormesh(K_PAR, K_PERP, lightcone_PS[i]['delta'] , cmap='BuPu', shading='auto', norm='log')
        colorbar = plt.colorbar(mesh, label=r'$\log_{10}(\Delta^2(k_\perp, k_\parallel))\left[\mathrm{mK^2}\right]$')
        plt.xlabel(r'$k_\perp \left[\mathrm{Mpc}^{-1}\right]$')
        plt.ylabel(r'$k_\parallel \left[\mathrm{Mpc}^{-1}\right]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.05, 1.1)
        plt.ylim(0.05, 1.1)
        plt.minorticks_off()
        plt.title(r'$\bar z$'f'={np.round(lightcone_redshifts[i],2)}')
        plt.grid(True)
        plt.show()



# computes 2D PS for a redshift chunk determined by z1 and z2, the default BOX_LEN is according to our standard LCs
def compute_2D_PS(file, z1, z2, BOX_LEN=200): # here file is .npz LC file
    field = read_image(file)
    DIM = field.shape[0]
    cell_size = BOX_LEN / DIM
    lc_redshifts = compute_redshifts(file)
    # Get indices that correspond to redshift range
    start = np.searchsorted(lc_redshifts, z1, side='left') 
    end = (np.searchsorted(lc_redshifts, z2, side='right') - 1 ) 
    chunklen = (end - start) * cell_size # the physical chuncklength in MPc
    print('physical chunklen:', np.round(chunklen, 2))
    # computing the PS
    k_perp, k_par, power = compute_2D_PS_chunk(field[:, :, start:end], BOX_LEN=BOX_LEN, DIM=DIM, chunklen=chunklen)
    # return power and compute mean redshift
    z_mean = np.mean(lc_redshifts[start:end])
    print('Mean redshift:', np.round(z_mean, 2))
    #xHI_mean.append(np.mean(field.xH_box[:,:,start:end]))
    #Tb_mean.append(np.mean(field.brightness_temp[:,:,start:end]))
    print('-----------------------------') 
    return k_perp, k_par, power, z1, z2



# plots the 2D PS computed by compute_2D_PS 
def plot_2D_PS(PS):
    from matplotlib.colors import LogNorm
    k_perp, k_par, power = PS[0], PS[1], PS[2]
    z1, z2 = PS[3], PS[4]
    # Create 2D plot (contour plot)
    K_PERP, K_PAR = np.meshgrid(k_perp, k_par)
    mesh = plt.pcolormesh(K_PERP, K_PAR, power, cmap='BuPu', shading='auto', norm=LogNorm(vmin=0.01, vmax=100.)) 
    colorbar = plt.colorbar(mesh, label=r'$\log_{10}(\Delta^2(k_\perp, k_\parallel))\left[\mathrm{mK^2}\right]$')
    #colorbar.set_clim(0, 2)
    #colorbar.set_ticklabels([0, 1, 2])

    # Label axes
    plt.xlabel(r'$k_\perp \left[\mathrm{Mpc}^{-1}\right]$')
    plt.ylabel(r'$k_\parallel \left[\mathrm{Mpc}^{-1}\right]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.05, 2.5)
    plt.ylim(0.05, 2.5)
    plt.minorticks_off()
    plt.title(f'21cm power spectrum for redshift range z = {z1,z2}')
    plt.grid(True)
    plt.show()



# Example 
# PS = compute_2D_PS(lc_3, z1=8.7, z2=10.6)
# plot_2D_PS(PS)

