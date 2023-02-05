import numpy as np


class arbt_moons:

    def __init__(self, n_moons=2, sigma=0.1, r=1, n_samples=100, moon_idadjust = 0.3):
        self.n_moons = n_moons
        self.sigma = sigma
        self.r = r
        self.n_samples = n_samples
        self.moon_idadjust = moon_idadjust
        self.moon_sz = int(self.n_samples / self.n_moons)
        self.moons = []


    def __call__(self):
        for moon_id in range(self.n_moons):
            q = np.random.uniform(0, np.pi, size=self.moon_sz)
            
            if moon_id % 2 == 0:
                factor = 1
            else: 
                factor = -1
            
            moon = np.zeros((self.moon_sz, 3))
            moon[:,0] = (self.r * np.cos(q)) + moon_id
            moon[:,1] = (self.r * np.sin(q) * factor) + (factor == -1) * self.moon_idadjust
            moon[:,2] = moon_id
            self.moons.append(moon)
            noise = np.random.normal(0, self.sigma, size=moon[:,:2].shape)
            moon[:,:2] += noise
        self.moons = np.concatenate(self.moons)

        return self.moons[:,:2], self.moons[:,2]

    def __iter__(self):
        return iter(self.__call__())