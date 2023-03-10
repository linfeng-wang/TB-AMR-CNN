python main.py -r gene.csv -o ERR6634978_oh ERR6634978_1.fastq.gz ERR6634978_2.fastq.gz


RUN micromamba install -n base -c bioconda -c conda-forge -y \
    trimmomatic=0.39 \
    bwa-mem2=2.2.1 \
    sambamba=0.6.8 \
    samtools=1.12 \
    pandas=1.4.2 && \
    micromamba clean --all --yes
