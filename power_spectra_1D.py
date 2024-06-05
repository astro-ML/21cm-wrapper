import numpy as np
import matplotlib.pyplot as plt
import py21cmfast as p21c

'''
# read some test cubes
lightcone = p21c.outputs.LightCone.read("./Lightcones/LightCone_z4.0.h5") # BOX_LEN = 300 , DIM = 150
coeval8 = p21c.outputs.Coeval.read("./Lightcones/Coeval_z8.0.h5")
#coeval10 = p21c.outputs.Coeval.read("./Lightcones/Coeval_z10.0.h5")
#coeval15 = p21c.outputs.Coeval.read("Coeval_z15.0.h5")
#lc_1 = './Lightcones/run895.npz' # BOX_LEN = 200 , DIM = 140
#lc_2 = './Lightcones/run996.npz'
lc_3 = './Lightcones/run795.npz'

gxH_redshifts = np.load('./Lightcones/redshifts5.npy')
'''

# reading the LCs and computing the redshifts (if loaded from a .npz)
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


'''
# plot gxH of our models to check reionisation history
plt.plot(gxH_redshifts, read_gxH(lc_1), label='1')
plt.plot(gxH_redshifts, read_gxH(lc_2), label='2')
plt.plot(gxH_redshifts, read_gxH(lc_3), label='3')
plt.plot(gxH_redshifts, read_gxH(lc_4), label='4')
plt.plot(gxH_redshifts, read_gxH(lc_5), label='5')
plt.plot(gxH_redshifts, read_gxH(lc_6), label='6')
plt.legend()
'''

# Computing the PS using powerbox
from powerbox.tools import get_power
def PS_powerbox(field):
    BOX_LEN = field.user_params.BOX_LEN 
    res, k = get_power(field.brightness_temp, boxlength=(BOX_LEN, BOX_LEN, BOX_LEN), vol_normalised_power=True)

    pow = k**3 * res / (2.0 * np.pi * np.pi )

    return k, pow


# building the 1D PS for coeval cube, directly takes the Coeval object
def compute_1D_PS_Coeval(field):
    DIM = field.user_params.HII_DIM # simulation grid
    BOX_LEN = field.user_params.BOX_LEN # physical dimension
    cell_size = BOX_LEN/DIM 
    print('resolution is:', cell_size)
    d_cell = cell_size/2/np.pi
    sample_rate = 1/d_cell
    DIM_MIDDLE = int(DIM/2+1)

    kbox = np.fft.rfftfreq(int(DIM), d=d_cell)
    #print('# of k bins: ', np.shape(kbox))

    # dim, binning etc
    k_factor = 1.4
    
    V = BOX_LEN**3

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
    
    print('NUM_BINS count',NUM_BINS)

    # initialise
    in_bin_ct = [0 for i in range(NUM_BINS)]
    p_box = [0 for i in range(NUM_BINS)]
    p_noise = [0 for i in range(NUM_BINS)]
    k_ave = [0 for i in range(NUM_BINS)]

    # take brightness_temp and scale field
    field_scaled = field.brightness_temp * cell_size**3  
    # fourier transform of field
    deldel_field = np.fft.rfftn(field_scaled) 
    field_amplitudes = np.abs(deldel_field)**2 

    # loop to construct Pk
    for n_x in range(DIM):
        if (n_x>DIM_MIDDLE):
            k_x =(n_x-DIM) * DELTA_K 
        else:
            k_x = n_x * DELTA_K
        for n_y in range(DIM):
            if (n_y>DIM_MIDDLE):  # if (n_x>DIM_MIDDLE):
                k_y =(n_y-DIM) * DELTA_K
            else:
                k_y = n_y * DELTA_K
            for n_z in range(DIM_MIDDLE):
                k_z = n_z * DELTA_K

                k_mag = np.sqrt(k_x*k_x + k_y*k_y + k_z*k_z)

                ct = 0
                k_floor = 0
                k_ceil = k_first_bin_ceil

                while k_ceil < k_max :
                    if ((k_mag>=k_floor) and (k_mag < k_ceil)):
                        in_bin_ct[ct] += 1
                        p_box[ct] += np.array(pow(k_mag,3)*field_amplitudes[n_x,n_y,n_z]/(2.0 * np.pi**2 * V)) 
                    
                        k_ave[ct] += k_mag
                        break
                    ct+=1
                    k_floor=k_ceil
                    k_ceil*=k_factor
                
    k_ave_norm = []
    p_box_norm = []
    p_noise_norm = []

    for ct in range(NUM_BINS):
        if ( in_bin_ct[ct]>0 and k_ave[ct]>0 ):
            k_ave_norm.append( k_ave[ct]/(in_bin_ct[ct]+0.0) )
            p_box_norm.append( p_box[ct]/(in_bin_ct[ct]+0.0) )


    return k_ave_norm, p_box_norm
    #plt.loglog(k_ave_norm, p_box_norm, label='PS from scratch', color='blue')



def plot_1D_PS_Coeval(coeval_PS):
    k, pow = coeval_PS
    plt.loglog(k, pow, label='PS')
    #plt.show()


# interpolate and compute ratio between 2 powerspectra (of unequal size)
def compare_PS(PS_small, PS_large):
    from scipy.interpolate import interp1d

    interpolator = interp1d(PS_small[0], PS_small[1], kind='cubic', fill_value='extrapolate')
    pow_me_interpolated = interpolator(PS_large[0])

    ratio = pow_me_interpolated / PS_large[1]

    # Plot the ratio
    plt.figure(figsize=(10, 6))
    plt.axhline(y=1, color='black')
    plt.semilogx(PS_large[0], ratio, label='pow_me / pow_package')
    plt.xlabel('k')
    plt.ylabel('Ratio (Self written / powerbox)')
    plt.title('PS comparison: Self written vs. powerbox')
    plt.ylim(0.8, 1.2)

    #plt.legend()
    plt.grid(True)
    plt.show()



# building the PS for Lightcone chunks
def compute_1D_PS_chunk(field, BOX_LEN, DIM, chunklen):  # field is already brightness_temp
    cell_size = BOX_LEN/DIM 
    print('resolution is:', np.round(cell_size, 2))
    d_cell = cell_size/2/np.pi
    sample_rate = 1/d_cell
    DIM_MIDDLE = int(DIM/2+1)

    kbox = np.fft.rfftfreq(int(DIM), d=d_cell)
    #print('# of k bins: ', np.shape(kbox))

    # dim, binning etc
    k_factor = 1.5
    
    V = BOX_LEN**2 * chunklen # Volume in Mpc**3

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
    
    print('NUM_BINS count',NUM_BINS)

    # initialise
    in_bin_ct = [0 for i in range(NUM_BINS)]
    p_box = [0 for i in range(NUM_BINS)]
    #p_noise = [0 for i in range(NUM_BINS)]
    k_ave = [0 for i in range(NUM_BINS)]

    # scale field
    field_scaled = field * cell_size**3
    # fourier transform of field
    deldel_field = np.fft.rfftn(field_scaled) 
    field_amplitudes = np.abs(deldel_field)**2 

    # loop to construct Pk
    for n_x in range(DIM):
        if (n_x>DIM_MIDDLE):
            k_x =(n_x-DIM) * DELTA_K 
        else:
            k_x = n_x * DELTA_K
        for n_y in range(DIM):
            if (n_y>DIM_MIDDLE):  # if (n_x>DIM_MIDDLE):
                k_y =(n_y-DIM) * DELTA_K
            else:
                k_y = n_y * DELTA_K
            for n_z in range(DIM_MIDDLE):
                k_z = n_z * DELTA_K

                k_mag = np.sqrt(k_x*k_x + k_y*k_y + k_z*k_z)

                ct = 0
                k_floor = 0
                k_ceil = k_first_bin_ceil

                while k_ceil < k_max :
                    if ((k_mag>=k_floor) and (k_mag < k_ceil)):
                        in_bin_ct[ct] += 1
                        p_box[ct] += np.array(pow(k_mag,3)*field_amplitudes[n_x,n_y,n_z]/(2.0 * np.pi**2 * V)) 
                    
                        k_ave[ct] += k_mag
                        break
                    ct+=1
                    k_floor=k_ceil
                    k_ceil*=k_factor
                
    k_ave_norm = []
    p_box_norm = []
    #p_noise_norm = []

    for ct in range(NUM_BINS):
        if ( in_bin_ct[ct]>0 and k_ave[ct]>0 ):
            k_ave_norm.append( k_ave[ct]/(in_bin_ct[ct]+0.0) )
            p_box_norm.append( p_box[ct]/(in_bin_ct[ct]+0.0) )


    return k_ave_norm, p_box_norm


# Now we devide lightcones into chunks and compute their PS.


def compute_PS_LC(field, nchunks=2): # here field is LC object
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
        print(chunklen)
        # computing the PS
        k, power = compute_1D_PS_chunk(field.brightness_temp[:, :, start:end], BOX_LEN=BOX_LEN, DIM=DIM, chunklen=chunklen)
        # append power and compute mean redshift, ionized fraction and brightness temp per chunk
        data.append({"k": k, "delta": power})
        z_mean.append(np.mean(field.lightcone_redshifts[start:end]))
        print('Mean redshift:', np.round(z_mean[i], 2))
        xHI_mean.append(np.mean(field.xH_box[:,:,start:end]))
        Tb_mean.append(np.mean(field.brightness_temp[:,:,start:end]))
        print('-----------------------------')
        
    return data, z_mean, xHI_mean, Tb_mean


# plots an lightcone_chunks object given the output of compute_PS_LC()
def plot_1D_PS_LC(lightcone_chunks):
    nchunks = len(lightcone_chunks[0])
    lightcone_PS = lightcone_chunks[0]
    lightcone_redshifts = lightcone_chunks[1]
    lightcone_ionized_frac = lightcone_chunks[2]
    lightcone_global_temp = lightcone_chunks[3]

    fig, axs = plt.subplots(nchunks, figsize=(8, 12), constrained_layout=True, sharex=True, sharey=True,subplot_kw={"xscale":'log', "yscale":'log'})
    plt.rcParams['text.usetex'] = True
    for i in range(nchunks):
        axs[i].plot(lightcone_PS[i]['k'], lightcone_PS[i]['delta'])
        #axs[i].text(0.12, 0.08, '$\overline z'f'={round(lightcone_redshifts[i],2)}$',horizontalalignment='right',verticalalignment='bottom', transform=axs[i].transAxes,fontsize = 12)
        #axs[i].text(0.99, 0.08, '$\overline x_{\mathrm{HI}}'f'={str(round(lightcone_ionized_frac[i], 2))}$',horizontalalignment='right',verticalalignment='bottom', transform=axs[i].transAxes,fontsize = 12)
        #axs[i].text(0.99, 0.18, '$\overline T_{\mathrm{b}}'f'={str(round(lightcone_global_temp[i], 2))}$',horizontalalignment='right',verticalalignment='bottom', transform=axs[i].transAxes,fontsize = 12)
        axs[i].set_ylabel(r'$\log_{10}(\Delta^2(k))\left[\mathrm{mK^2}\right]$')
        #axs[i].set_ylim(2e-3, 2e2)
        #axs[i].set_xlim(3e-2, 2)
        
    fig.suptitle('21cm power spectrum')
    plt.xlabel('$k$')
    #plt.ylabel('$k^3 P$')
    #plt.savefig(fname='./Plots/PS_z4_ave.pdf')
    plt.show()


# Now for general arrays like the LCs on the cluster:


# computes PS for given array and redshift slice
def compute_1D_PS(file, z1, z2, BOX_LEN=200): # here file is .npz LC file
    field = read_image(file)
    DIM = field.shape[0]
    cell_size = BOX_LEN / DIM
    lc_redshifts = compute_redshifts(file)
    # Get indices that correspond to redshift range
    start = np.searchsorted(lc_redshifts, z1, side='left') 
    end = (np.searchsorted(lc_redshifts, z2, side='right') - 1 ) 
    chunklen = (end - start) * cell_size # the physical chuncklength in MPc
    print('physical chunklen:', chunklen)
    # computing the PS
    k, power = compute_1D_PS_chunk(field[:, :, start:end], BOX_LEN=BOX_LEN, DIM=DIM, chunklen=chunklen)
    # return power and compute mean redshift
    z_mean = np.mean(lc_redshifts[start:end])
    print('Mean redshift:', np.round(z_mean, 2))
    #xHI_mean.append(np.mean(field.xH_box[:,:,start:end]))
    #Tb_mean.append(np.mean(field.brightness_temp[:,:,start:end]))
    print('-----------------------------') 
    return k, power, z1, z2


# plots the 1D PS for a redshift chunk determined by z1 and z2, the default BOX_LEN is according to our standard LCs
def plot_1D_PS(PS):
    k, power = PS[0], PS[1]
    z1, z2 = PS[2], PS[3]
    plt.loglog(k, power)
    plt.ylim(0.1, 100)
    plt.ylabel(r'$\Delta^2 (k) [\mathrm{mK^2}]$')
    plt.xlabel(r'$k[\mathrm{Mpc^{-1}}]$')
    #plt.ylim(10 ** np.floor(np.log10(np.min(PS[1]))), 10 ** np.ceil(np.log10(np.max(PS[1]))))
    plt.minorticks_off()
    plt.title(f'21cm power spectrum for redshift range z = {z1,z2}')
    plt.show()



# Example 
# PS = compute_1D_PS(lc_3, z1=7.7, z2=8.6)
# plot_1D_PS(PS)

