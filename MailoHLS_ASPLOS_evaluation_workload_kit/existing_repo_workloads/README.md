# Existing evaluation workloads

The locked set also contains **serrano-kalman-filter** and **GAN**. Keep their
current sources/preprocessing from the `ElenaVouvali/MailoHLS`
`stage2-analysis-refactor` branch rather than duplicating them here; this avoids
creating a stale fork of the two workload inputs that are already maintained with
the framework.

Kalman is the frozen held-out test family. GAN is an external application. The
remaining seven external inputs are supplied/prepared by the sibling directories.
