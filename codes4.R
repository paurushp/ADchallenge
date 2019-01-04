# 1 ## Via Biomart

> myMart=useMart("unimart", dataset="uniprot")
> myMart
> listAttributes(myMart)
> myProt=getBM(attributes=c("name", "go_id", "ensembl_id"), filter="accession", values="Q13613", mart=myMart)
> myProt
         name      go_id
1 MTMR1_HUMAN GO:0005829
2 MTMR1_HUMAN GO:0005886
3 MTMR1_HUMAN GO:0004725
4 MTMR1_HUMAN GO:0044281
5 MTMR1_HUMAN GO:0035335
6 MTMR1_HUMAN GO:0004438
7 MTMR1_HUMAN GO:0006661

> myProtein=getSequence(id=myProt$ensembl_id[1],type="ensembl_gene_id", seqType="peptide", mart=mart) 

# 2 ## Via uniprot.ws


> availableUniprotSpecies(pattern="sapiens")
  taxon ID                  Species name
1    63221 Homo sapiens neanderthalensis
2     9606                  Homo sapiens
> taxId(UniProt.ws)=9606

> head(keytypes(UniProt.ws))
[1] "UNIPROTKB"         "UNIPARC"           "UNIREF50"         
[4] "UNIREF90"          "UNIREF100"         "EMBL/GENBANK/DDBJ"

# via protr
ids = c(’P00750’, ’P00751’, ’P00752’)
getUniProt(ids)

> res2=select(UniProt.ws, keys=c("P31946","P62258"), cols=c("PDB","SEQUENCE"),
keytype="UNIPROTKB")

> keys=c("1","10","1000")
> cols=c("PDB", "UNIGENE", "SEQUENCE")
> kt="ENTREZ_GENE"
> res=select(UniProt.ws, keys, cols, kt)

# 3 ##interpro domain

> refseqids = c("NM_005359","NM_000546")
> ipro = getBM(attributes=c("refseq_dna","interpro","interpro_description"), filters="refseq_dna",values=refseqids, mart=ensembl)
interpro
refseq_dna interpro interpro_description

1 NM_000546 IPR002117 p53 tumor antigen
2 NM_000546 IPR010991 p53, tetramerisation
3 NM_000546 IPR011615 p53, DNA-binding
4 NM_000546 IPR013872 p53 transactivation domain (TAD)
5 NM_000546 IPR000694 Proline-rich region
6 NM_005359 IPR001132 MAD homology 2, Dwarfin-type
7 NM_005359 IPR003619 MAD homology 1, Dwarfin-type
8 NM_005359 IPR013019 MAD homology

# 4 ## Handling PDB file
library(bio3d)
pdb1 <- read.pdb("1bg2")
summary(pdb1)
str(pdb1)



# 4 ## Protein Sequence analysis Muscle should be installed  sudo apt-get install muscle


pdb1 <- read.pdb("1bg2")
pdb2 <- read.pdb("4hna")
pdb3 <- read.pdb("1mkj")
s1= aa321(pdb1$seqres)
s2= aa321(pdb2$seqres)
s3= aa321(pdb3$seqres)
aln <- seqaln( seqbind(s1, s2,s3), id=c("1bg2","4hna", "1mkj") )

> prots=c("1bg2", "4hna", "1mkj")
> raw <- NULL
    for(i in 1:length(prots)) {
      pdb <- read.pdb(prots[i])
      raw <- seqbind(raw, c(aa321(pdb$atom[pdb$calpha,"resid"])))
    }
> aln <- seqaln(raw, id=prots), file="seqal.fa")
> plot.bio3d( conserv(aln)[!is.gap(aln$ali[1,])])
> aln2html(aln, append=FALSE, file="myeg.html")

# 5 ## Sequence Feature computation

library(protr)
s1=paste(s1, sep="",collapse="")
extractAAC(s1)
extractAPAAC(s1, props = c("Hydrophobicity", "Hydrophilicity"),lambda = 30, w = 0.05, customprops = NULL)
extractCTDC(s1)
extractCTDD(s1)
extractDC(s1)


extractPAAC(s1, props = c("Hydrophobicity", "Hydrophilicity", "SideChainMass"), lambda = 30, w = 0.05, customprops = NULL)

protcheck(s1)

# 6 ## Ramachandran plot
library(bio3d)
tor=torsion.pdb(pdb)
plot(tor$phi, tor$psi)


scatter.psi <- tor$psi
scatter.phi <- tor$phi
par(pty="s")
plot(x=scatter.phi, y=scatter.psi, xlim=c(-180,180), ylim=c(-180,180), main="General", xlab=expression(phi), ylab=expression(psi))

par(mfrow=c(2,2)
grid.filename <- paste(grid.dir, grid.filenames[rama.type], sep="")
grid <- load.grid(grid.filename, mid.points)
par(mar=c(3,3,3,3), mgp=c(1.75,0.75,0), pty="s")

    ramachandran.plot(scatter.phi, scatter.psi,
             x.grid=mid.points, y.grid=mid.points, z.grid=grid,
             plot.title=col.name,
             levels=grid.levels[rama.type,],
             col=grid.colors[rama.type,])
}

# 7 Similar protein search


pdb <- read.pdb("4q21")            
blast <- blast.pdb( seq.pdb(pdb) )  
head(blast$hit.tbl)                   
top.hits <- plot(blast)                     
head(top.hits$hits)                   

  pdb.id   gi.id       group
1 "6Q21_A" "231226"    "1"  
2 "6Q21_B" "231227"    "1"  
3 "6Q21_C" "231228"    "1"  
4 "6Q21_D" "231229"    "1"  
5 "1IOZ_A" "15988032"  "1"  
6 "1AA9_A" "157829765" "1"  

# 7 Similar structure search




### 8.  secondary structure
sudo wget ftp://ftp.cmbi.ru.nl/pub/software/dssp/dssp-2.0.4-linux-i386 -O /usr/local/bin/dssp

sudo chmod a+x /usr/local/bin/dssp

SS <- dssp(pdb,  exepath ="/usr/local/bin")
#
library(Rknots)
protein=loadProtein("4q21")

protein <- newKnot(protein$A)
protein.cp <- closeAndProject(protein)
# Plot the results
par(mfrow = c(1,2))
plot(protein, lwd = 2)
plot(protein.cp, lwd = 2)


plot3D(protein, radius = 0.1)
plot3D(protein, radius = 0.1, lwd = 10)
plot3D(protein)

plotKnot3D(protein, ends = c(), text = FALSE)
link <- AlexanderBriggs(protein$points3D, protein$ends)
plotDiagram(link$points3D, link$ends, main = i, lwd = 1.5)
plotDiagram(protein)


par(mfrow = c(1,2))
plotDiagram(protein$A, ends = c(), lwd = 2.5, main = ’Original’)
protein.rot <- PCAProjection(protein$A)
plotDiagram(protein.rot, ends = c(), lwd = 2.5, main = ’Reduced’)

