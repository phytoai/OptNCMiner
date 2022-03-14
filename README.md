#### Collecting Data for T2DM

### 1. download

[Downloads](https://foodb.ca/downloads)

- download FooDB MySQL Dump file

---

### 2. import dump file

- decompress foodb_2020_4_7_mysql.tar.gz
- create db

```sql
CREATE DATABASE foodb;
```

- import dump file

```sql
mysql -u [userid] -p [password] < {path/foodb_server_dump_2020_4_21}.sql
```

---

### 3. extract data

- search smiles

```sql
USE foodb;

SELECT DISTINCT c.moldb_smiles FROM 
contents t JOIN compounds c ON t.source_id = c.id 
JOIN foods f ON t.food_id = f.id
WHERE t.source_type='Compound' and c.export = 1 and f.export_to_foodb = 1
```

- download csv file ( moldb_smiles.csv )

---

### 4. add fingerprint

```r
#install.packages('caret')
#install.packages('rcdk')
library(caret)
library(rcdk)

data<-read.csv("moldb_smiles.csv")
dim(data)
  
# extract SMILES column
data<-as.character(data[,1])
 
# parsing
mols = parse.smiles(data)

trans <- data.frame(data[!sapply(mols, is.null)], stringsAsFactors=F)
mols = parse.smiles(trans[,1])

fps = lapply(mols, get.fingerprint, type='standard')
fps.matrix = fingerprint::fp.factor.matrix(fps)

# save data
final <- data.frame(cbind(trans, fps.matrix))
write.csv(final, "foodb_compounds_fp.csv",row.names=FALSE)
```

---

### 5. rename columns

- data..sapply.mols..is.null.. → SMILES
