import pandas as pd
import matplotlib.pyplot as plt

class S2DataLoader:
    def __init__(self):
        self.astro_df = None
        self.vel_df = None
    
    def load_data(self):
        # Load astrometry
        astro_naco = pd.read_csv('data/astrometry_NACO.csv')
        astro_sharp = pd.read_csv('data/astrometry_SHARP.csv')
        self.astro_df = pd.concat([astro_naco, astro_sharp]).dropna()
        
        # Load velocity
        vel_naco = pd.read_csv('data/velocity_NACO.csv')
        vel_osiris = pd.read_csv('data/velocity_OSIRIS.csv')
        vel_sinfoni = pd.read_csv('data/velocity_SINFONI.csv')
        self.vel_df = pd.concat([vel_naco, vel_osiris, vel_sinfoni]).dropna()
        
        return self.astro_df, self.vel_df
    
    def get_data(self):
        if self.astro_df is None or self.vel_df is None:
            raise RuntimeError("Call load_data() first")
        return {
            't_ast': self.astro_df['t'].values,
            'ra_obs': self.astro_df['x'].values,
            'dec_obs': self.astro_df['y'].values,
            'ra_err': self.astro_df['x_err'].values,
            'dec_err': self.astro_df['y_err'].values,
            't_rv': self.vel_df['t'].values,
            'rv_obs': self.vel_df['vz'].values,
            'rv_err': self.vel_df['vz_err'].values
        }
    
    def plot_all(self, figsize=(12, 10)):
        if self.astro_df is None or self.vel_df is None:
            raise RuntimeError("Load data first using load_data()")
            
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        sc = axs[0,0].scatter(self.astro_df['x'], self.astro_df['y'], 
                              c=self.astro_df['t'], cmap='viridis', s=20)
        axs[0,0].errorbar(self.astro_df['x'], self.astro_df['y'],
                          xerr=self.astro_df['x_err'], yerr=self.astro_df['y_err'],
                          fmt='none', ecolor='black', alpha=0.3)
        plt.colorbar(sc, ax=axs[0,0], label='Time (year)')
        axs[0,0].set(xlabel='RA (arcsec)', ylabel='Dec (arcsec)', title='Sky Position')
        
        axs[0,1].errorbar(self.vel_df['t'], self.vel_df['vz'],
                          yerr=self.vel_df['vz_err'],
                          fmt='o', color='red', markersize=4, alpha=0.7)
        axs[0,1].set(xlabel='Time (year)', ylabel='RV (km/s)', title='Radial Velocity')
        
        axs[1,0].errorbar(self.astro_df['t'], self.astro_df['x'],
                          yerr=self.astro_df['x_err'],
                          fmt='o', color='blue', markersize=4, alpha=0.7)
        axs[1,0].set(xlabel='Time (year)', ylabel='RA (arcsec)', title='RA Motion')
        
        axs[1,1].errorbar(self.astro_df['t'], self.astro_df['y'],
                          yerr=self.astro_df['y_err'],
                          fmt='o', color='green', markersize=4, alpha=0.7)
        axs[1,1].set(xlabel='Time (year)', ylabel='Dec (arcsec)', title='Dec Motion')
        
        plt.tight_layout()
        return fig
