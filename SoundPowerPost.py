import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fileinp = 'LAEQ'
df_in = pd.read_csv("./" + fileinp + ".csv")

# Convert DataFrame to 2D NumPy array
data_array = df_in.values

freq = np.array([6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])

# Separate measured sound and background noise
background_noise= data_array[:5, :]
measured_sound  = data_array[5:10, :]

def mean(dBdata):
    Nm = len(dBdata[:, 0])
    
    sumdB = np.zeros_like(dBdata[0, :])

    for i in range(Nm):
        sumdB += np.power(10, 0.1 * dBdata[i, :])

    sumdB = sumdB / Nm
    meandB = 10 * np.log10(sumdB)

    return meandB

# Calculate means of the whole spectrum
LpaSource_mean = mean(measured_sound)
LpaBackgroud_mean = mean(background_noise)

# Find the index where Lpa is positive
# to find index where Lpa is negative 

for i in range (len(LpaSource_mean)):
    if (LpaSource_mean[i] > 0):
        minIndex = i+1
        break

# Truncate the spectrum
LpaSource_mean = LpaSource_mean[minIndex:]
LpaBackgroud_mean = LpaBackgroud_mean[minIndex:]
freq = freq[minIndex:]

print('LpaSource_mean after truncation: ')
print(LpaSource_mean)
print('LpaBackground_mean after truncation: ')
print(LpaBackgroud_mean)
print('Frequencies after truncation: ')
print(freq)

# Calculate deltaLpa with a small positive number epsilon
epsilon = 1e-6
deltaLpa = np.abs(LpaSource_mean - LpaBackgroud_mean)
deltaLpa = np.clip(deltaLpa, a_min=epsilon, a_max=None)


print('DELLPA;', deltaLpa)

    
# Calculate k1A
k1A = -10 * np.log10(1 - np.power(10, -0.1 * deltaLpa))

for i in range (len(deltaLpa)):
    if (deltaLpa[i] > 10):
        k1A[i] = 0
    elif(deltaLpa[i] < 3):
        k1A[i] = 3

alpha = 0.15
S = (8.98/2) + 4.641 + 4.6852
Sv = 91 + 52 + 56 
A = alpha * Sv

# Calculate k2A
k2A = 10 * np.log10(1 + 4 * (S/A))
print('####')
print('k1A', k1A,'k2A', k2A)
print('####')
# Calculate LpaMean_Source_final
LpaMean_Source_final = LpaSource_mean - k1A - k2A  # Fix: use k2A instead of hardcoded constant

S0 = 1
Lwa = LpaMean_Source_final + 10 * np.log10(S/S0)

print('LpaMean_Source_final:', LpaMean_Source_final)
print('Lwa:', Lwa)

# Choose scientific colors for the lines
color_lpa = 'b'  # Blue
color_lwa = 'r'  # Red

# Plot the results
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(freq, LpaMean_Source_final, label='LpA', linestyle='-', marker='o', color=color_lpa)
plt.plot(freq, Lwa, label='LwA', linestyle='--', marker='x', color=color_lwa)
plt.xscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)plt.xlabel('Frequency (Hz)')
plt.ylabel('Sound Pressure Level (dB)')
plt.legend()
plt.tight_layout()  # Improves spacing
plt.savefig('sound.png', dpi=500)
plt.show()
# Create a DataFrame with the desired data
result_data = {
    'frequency': freq,
    'LpA_': LpaSource_mean,
    'LPaB': LpaBackgroud_mean,
    'delLpA': deltaLpa,
    'K1A': k1A,
    'LpA': LpaMean_Source_final,
    'LWA': Lwa
}

result_df = pd.DataFrame(result_data)

# Save DataFrame to CSV file
result_df.to_csv('output_data.csv', index=False)

print('CSV file generated successfully.')
