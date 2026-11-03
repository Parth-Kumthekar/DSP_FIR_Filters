'''Instead of using bandstop filters to remove noise use an adaptive LMS filter by providing it
with a noise reference. Add an adaptive LMS filter command to your FIR filter class and
name it: “doFilterAdaptive(self,signal,noise,learningRate)” which returns the cleaned-up
ECG. As before this function must also receive only scalars (i.e. sample by sample) and
return a scalar. Plot and compare the result from the adaptive filter and that from the FIR
filter design'''