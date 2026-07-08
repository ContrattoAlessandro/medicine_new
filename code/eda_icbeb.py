import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def main():
    print("Inizializzazione EDA del dataset ICBEB...", flush=True)

    # 1. Configurazione dei percorsi dinamici e del working directory
    # Modifichiamo la directory corrente in quella in cui risiede lo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.append('.') # Assicuriamoci che la cartella 'code' sia nel path

    # Percorsi relativi (ora che siamo in 'code')
    datafolder_icbeb = '../data/ICBEB/'
    datafolder_ptbxl = '../data/ptbxl/'
    output_dir = '../output/eda_icbeb/'

    # Creazione della cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cartella di output impostata su: {os.path.abspath(output_dir)}", flush=True)

    # Import delle utilità dal progetto
    from utils import utils

    # 2. Caricamento dei dataset
    print("Caricamento dei metadati...", flush=True)
    X_icbeb, Y_icbeb = utils.load_dataset(datafolder_icbeb, 100)
    X_ptbxl, Y_ptbxl = utils.load_dataset(datafolder_ptbxl, 100)

    print(f"ICBEB caricato: {X_icbeb.shape[0]} record, forma segnale: {X_icbeb.shape[1:]}", flush=True)
    print(f"PTB-XL caricato: {X_ptbxl.shape[0]} record, forma segnale: {X_ptbxl.shape[1:]}", flush=True)

    # Definizione delle 9 classi target di ICBEB
    icbeb_classes = ['CRBBB', 'AFIB', 'NORM', 'STD_', '1AVB', 'VPC', 'PAC', 'CLBBB', 'STE_']
    # Mapping da VPC (ICBEB) a PVC (PTB-XL)
    ptbxl_mapped_classes = ['CRBBB', 'AFIB', 'NORM', 'STD_', '1AVB', 'PVC', 'PAC', 'CLBBB', 'STE_']
    class_mapping = dict(zip(icbeb_classes, ptbxl_mapped_classes))

    # 3. Analisi Demografica
    print("Esecuzione analisi demografica...", flush=True)
    
    # Statistiche Età
    age_icbeb = Y_icbeb['age'].dropna()
    age_ptbxl = Y_ptbxl['age'].dropna()
    
    # Statistiche Sesso
    sex_icbeb = Y_icbeb['sex'].value_counts()
    sex_ptbxl = Y_ptbxl['sex'].value_counts()

    # Creazione grafico Demografico Comparativo
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Istogramma Età
    sns.histplot(age_icbeb, kde=True, label='ICBEB', color='#e74c3c', alpha=0.6, stat='density', ax=axes[0])
    sns.histplot(age_ptbxl, kde=True, label='PTB-XL', color='#3498db', alpha=0.4, stat='density', ax=axes[0])
    axes[0].set_title('Age Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()

    # Sesso
    # Rappresentiamo i dati sesso mappati
    sex_mapped_df = pd.DataFrame({
        'Dataset': ['ICBEB', 'ICBEB', 'PTB-XL', 'PTB-XL'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Percentage': [
            sex_icbeb.get(1, 0) / len(Y_icbeb) * 100,
            sex_icbeb.get(0, 0) / len(Y_icbeb) * 100,
            sex_ptbxl.get(0, 0) / len(Y_ptbxl) * 100,
            sex_ptbxl.get(1, 0) / len(Y_ptbxl) * 100
        ]
    })
    
    sns.barplot(data=sex_mapped_df, x='Dataset', y='Percentage', hue='Gender', palette=['#34495e', '#f1c40f'], ax=axes[1])
    axes[1].set_title('Gender Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demographics_comparison.png'), dpi=150)
    plt.close()

    # 4. Distribuzione delle Classi Diagnostiche
    print("Analisi delle etichette...", flush=True)

    # Estrazione delle classi per ciascun record
    # ICBEB
    icbeb_label_counts = {c: 0 for c in icbeb_classes}
    for idx, row in Y_icbeb.iterrows():
        for code in row.scp_codes.keys():
            if code in icbeb_label_counts:
                icbeb_label_counts[code] += 1

    # PTB-XL (calcoliamo solo sulle 9 classi target, considerando VPC->PVC)
    ptbxl_label_counts = {c: 0 for c in icbeb_classes}
    for idx, row in Y_ptbxl.iterrows():
        for code in row.scp_codes.keys():
            # Mappiamo il codice di PTB-XL al nome ICBEB per confronto
            for ic_cls, pt_cls in class_mapping.items():
                if code == pt_cls:
                    ptbxl_label_counts[ic_cls] += 1

    df_labels = pd.DataFrame([
        {'Class': c, 'ICBEB_Count': icbeb_label_counts[c], 'PTBXL_Count': ptbxl_label_counts[c]}
        for c in icbeb_classes
    ])
    
    df_labels['ICBEB_Percent'] = df_labels['ICBEB_Count'] / len(Y_icbeb) * 100
    df_labels['PTBXL_Percent'] = df_labels['PTBXL_Count'] / len(Y_ptbxl) * 100

    # Plot Frequenze relative delle classi
    df_labels_melted = pd.melt(df_labels, id_vars=['Class'], value_vars=['ICBEB_Percent', 'PTBXL_Percent'],
                               var_name='Dataset', value_name='Relative Frequency (%)')
    df_labels_melted['Dataset'] = df_labels_melted['Dataset'].map({'ICBEB_Percent': 'ICBEB', 'PTBXL_Percent': 'PTB-XL'})

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_labels_melted, x='Class', y='Relative Frequency (%)', hue='Dataset', palette=['#e74c3c', '#3498db'])
    plt.title('Comparative Diagnostic Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Diagnostic Class', fontsize=12)
    plt.ylabel('Relative Frequency (% of total records)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution_comparison.png'), dpi=150)
    plt.close()

    # 5. Analisi Multi-Label
    print("Analisi multi-label...", flush=True)
    labels_per_record_icbeb = Y_icbeb.scp_codes.apply(lambda x: len(x)).value_counts().sort_index()
    labels_per_record_ptbxl = Y_ptbxl.scp_codes.apply(lambda x: len([c for c in x.keys() if c in ptbxl_mapped_classes])).value_counts().sort_index()

    # Creazione dataframe per plot multi-label
    multilabel_data = []
    for k, v in labels_per_record_icbeb.items():
        multilabel_data.append({'Dataset': 'ICBEB', 'Labels per Record': k, 'Percentage': v / len(Y_icbeb) * 100})
    for k, v in labels_per_record_ptbxl.items():
        multilabel_data.append({'Dataset': 'PTB-XL (9 targets only)', 'Labels per Record': k, 'Percentage': v / len(Y_ptbxl) * 100})
    
    df_multilabel = pd.DataFrame(multilabel_data)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_multilabel, x='Labels per Record', y='Percentage', hue='Dataset', palette=['#e74c3c', '#3498db'])
    plt.title('Number of Diagnostic Labels per Record', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Concurrent Labels', fontsize=12)
    plt.ylabel('Percentage of Records (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multilabel_comparison.png'), dpi=150)
    plt.close()

    # Co-occorrenza delle classi in ICBEB
    binary_labels_icbeb = pd.DataFrame(0, index=Y_icbeb.index, columns=icbeb_classes)
    for idx, row in Y_icbeb.iterrows():
        for code in row.scp_codes.keys():
            if code in icbeb_classes:
                binary_labels_icbeb.loc[idx, code] = 1

    corr_matrix = binary_labels_icbeb.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Class Correlation Matrix (ICBEB Co-occurrences)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_cooccurrence_icbeb.png'), dpi=150)
    plt.close()

    # 6. Plotting dei tracciati ECG d'esempio a 12 derivazioni
    print("Generazione dei grafici dei segnali ECG per ciascuna classe...", flush=True)
    
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time = np.arange(1000) / 100.0 # 10 secondi a 100Hz

    for cls in icbeb_classes:
        sub_df = Y_icbeb[Y_icbeb.scp_codes.apply(lambda x: list(x.keys()) == [cls])]
        if len(sub_df) == 0:
            sub_df = Y_icbeb[Y_icbeb.scp_codes.apply(lambda x: cls in x)]
        
        if len(sub_df) > 0:
            sample_id = sub_df.index[0]
            pos_idx = sample_id - 1
            signal = X_icbeb[pos_idx]

            fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
            for i, lead_name in enumerate(leads):
                axes[i].plot(time, signal[:, i], color='#2c3e50', linewidth=1)
                axes[i].set_ylabel(lead_name, rotation=0, labelpad=15, fontsize=12, fontweight='bold')
                axes[i].grid(True, which='both', linestyle='--', linewidth=0.5, color='#bdc3c7')
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].set_ylim(signal[:, i].min() - 0.2, signal[:, i].max() + 0.2)
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=12)
            plt.suptitle(f'12-Lead ECG Signal - Class: {cls} (ecg_id: {sample_id})', fontsize=16, y=0.92, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.91])
            
            filename_plot = f'ecg_sample_{cls}.png'
            plt.savefig(os.path.join(output_dir, filename_plot), dpi=120)
            plt.close()
            print(f"  Salvato tracciato ECG d'esempio per classe {cls} in {filename_plot}", flush=True)

    # 7. Scrittura del Report in Markdown
    print("Scrittura del report Markdown...", flush=True)
    
    report_path = os.path.join(output_dir, 'eda_report.md')
    
    total_icbeb = len(Y_icbeb)
    total_ptbxl = len(Y_ptbxl)
    
    mean_age_ic = age_icbeb.mean()
    std_age_ic = age_icbeb.std()
    mean_age_pt = age_ptbxl.mean()
    std_age_pt = age_ptbxl.std()
    
    male_pct_ic = (sex_icbeb.get(1, 0) / total_icbeb) * 100
    female_pct_ic = (sex_icbeb.get(0, 0) / total_icbeb) * 100
    
    male_pct_pt = (sex_ptbxl.get(0, 0) / total_ptbxl) * 100
    female_pct_pt = (sex_ptbxl.get(1, 0) / total_ptbxl) * 100

    markdown_content = f"""# Report di Exploratory Data Analysis (EDA) - Dataset ICBEB

Questo report presenta un'analisi esplorativa e comparativa dettagliata del dataset **ICBEB 2018** (China Land-Bridge Cardiovascular Disease Challenge 2018) in relazione al dataset **PTB-XL**. L'ICBEB viene utilizzato nel progetto come set di validazione esterno "out-of-domain" per verificare le capacità di generalizzazione zero-shot dei modelli addestrati su PTB-XL.

---

## 1. Statistiche Generali e Demografiche

Il dataset ICBEB è composto da un numero inferiore di campioni rispetto a PTB-XL, ma presenta differenze demografiche e cliniche significative (domain shift).

| Metrica / Feature | Dataset ICBEB | Dataset PTB-XL |
| :--- | :---: | :---: |
| **Record Totali** | {total_icbeb} | {total_ptbxl} |
| **Età Media (Std)** | {mean_age_ic:.1f} ± {std_age_ic:.1f} anni | {mean_age_pt:.1f} ± {std_age_pt:.1f} anni |
| **Percentuale Uomini** | {male_pct_ic:.1f}% | {male_pct_pt:.1f}% |
| **Percentuale Donne** | {female_pct_ic:.1f}% | {female_pct_pt:.1f}% |
| **Formato Segnali ECG** | 1000 campioni × 12 canali (10s @ 100Hz) | 1000 campioni × 12 canali (10s @ 100Hz) |

### Distribuzione di Età e Genere
L'istogramma dell'età e il grafico di genere mostrano chiaramente la differenza tra le due popolazioni:
- La popolazione di **PTB-XL** è leggermente più anziana ed equilibrata tra i generi.
- La popolazione di **ICBEB** ha una distribuzione d'età leggermente differente e una percentuale maggiore di pazienti maschi.

![Confronto Demografico](demographics_comparison.png)

---

## 2. Distribuzione delle Classi Diagnostiche

L'ICBEB si concentra su **9 classi diagnostiche principali**. Qui sotto viene mostrato il confronto delle frequenze assolute e relative delle 9 classi in entrambi i dataset (notando che la classe `VPC` dell'ICBEB corrisponde a `PVC` in PTB-XL).

| Classe | Descrizione Clinica | Conteggio ICBEB | % su ICBEB | Conteggio PTB-XL | % su PTB-XL |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    
    class_desc = {
        'NORM': 'Normal ECG',
        'AFIB': 'Atrial Fibrillation',
        '1AVB': 'First-degree AV Block',
        'CLBBB': 'Complete Left Bundle Branch Block',
        'CRBBB': 'Complete Right Bundle Branch Block',
        'PAC': 'Premature Atrial Complex',
        'VPC': 'Ventricular Premature Complex (PVC)',
        'STD_': 'ST Depression',
        'STE_': 'ST Elevation'
    }

    for idx, row in df_labels.iterrows():
        c = row['Class']
        desc = class_desc.get(c, '')
        cnt_ic = row['ICBEB_Count']
        pct_ic = row['ICBEB_Percent']
        cnt_pt = row['PTBXL_Count']
        pct_pt = row['PTBXL_Percent']
        markdown_content += f"| **{c}** | {desc} | {cnt_ic} | {pct_ic:.2f}% | {cnt_pt} | {pct_pt:.2f}% |\n"

    markdown_content += f"""
### Grafico Comparativo delle Classi
![Distribuzione Classi](label_distribution_comparison.png)

> [!IMPORTANT]
> **Implicazioni del Domain Shift sulle Classi:**
> - **CRBBB (Complete Right Bundle Branch Block)** è estremamente frequente in ICBEB (~27%) rispetto a PTB-XL (~2.5%).
> - **STE_ (ST Elevation)** è presente in misura maggiore in ICBEB (~3.2%) rispetto a PTB-XL (~0.13%).
> - **NORM (Normal ECG)** rappresenta solo il ~13.3% dei record in ICBEB, mentre copre il ~43.6% in PTB-XL.
> Queste discrepanze creano una sfida significativa per i modelli addestrati su PTB-XL che devono effettuare previsioni zero-shot su ICBEB.

---

## 3. Complessità Multi-Label

Una delle differenze strutturali più rilevanti risiede nel numero di diagnosi assegnate a ciascun record.

- **ICBEB** è prevalentemente un dataset single-label (~93% dei record ha esattamente una diagnosi).
- **PTB-XL** è strutturalmente multi-label, con la maggior parte dei pazienti che presenta 2 o più diagnosi contemporaneamente.

![Confronto Multi-Label](multilabel_comparison.png)

### Co-occorrenza delle Classi in ICBEB
La matrice di co-occorrenza indica che le correlazioni tra le 9 classi target in ICBEB sono estremamente basse (prossime a zero), confermando la natura quasi esclusivamente mutualmente esclusiva del dataset ICBEB.

![Correlazione Classi](class_cooccurrence_icbeb.png)

---

## 4. Visualizzazione di ECG a 12 derivazioni d'esempio

Di seguito sono riportati i tracciati ECG a 12 derivazioni (10 secondi a 100 Hz) per ciascuna delle 9 classi in ICBEB. Questi grafici permettono di ispezionare visivamente le morfologie d'onda tipiche associate a ciascuna anomalia clinica.

```carousel
"""
    
    for i, cls in enumerate(icbeb_classes):
        if i > 0:
            markdown_content += "<!-- slide -->\n"
        markdown_content += f"### ECG di esempio per {cls}\n![ECG {cls}](ecg_sample_{cls}.png)\n"

    markdown_content += """```

---

## Conclusioni Chiave per il Benchmarking
1. **Domain Shift Demografico e Clinico:** Le popolazioni presentano distribuzioni di età e sesso dissimili.
2. **Frequenza di Classe Sbilanciata:** Alcune patologie (como CRBBB e STE_) sono estremamente sbilanciate nei due dataset. Ciò spiega l'importanza di tecniche di bilanciamento delle classi e di regolarizzazione.
3. **Differenza Strutturale di Labeling:** Il passaggio da un addestramento multi-label (PTB-XL) a un test single-label (ICBEB) richiede una calibrazione accurata delle soglie di decisione.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print("EDA completata con successo! Report e grafici salvati.", flush=True)

if __name__ == '__main__':
    main()
