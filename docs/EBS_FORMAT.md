# EEG Binary Stream (EBS) v0.1
* Little-endian, 32-bit float samples
* Header – 512 bytes JSON:
  { "sfreq":512,"channels":["F3","F4",...], "window_samples":3584, "uuid":"..." }
* Data – consecutive windows, interleaved by channel.
Compatibility: mmap-friendly, directly loadable by NumPy (`dtype='<f4'`).  
Scala & Python writers/readers will be added in later phases.
