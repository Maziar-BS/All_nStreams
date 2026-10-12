# All_nStreams
_Project Definition_: CUDA-based project developed in order to invoke benchmarks from NVIDIA Sample into a single platform for streaming purpose. The project takes input sizes and runtime measures from a command line and combines the inputs in every possible combination. Each Combination has a duration attached to it which is reported after each execution.

_Considerations_: Originally Developed in IDE : Microsoft Visual Studio 2017 -- NVIDIA CUDA Toolkit v.10.0

_Tutorial_: Creating a visual studio project by having *.sln file in the main repository and putting the other branch files in All_nStream Folder. Visual Studio based project forms as : 
1) https://drive.google.com/file/d/1MQmqIfKydqF--ohDagYnSzhUKpr4MlqS/view?usp=sharing
2) https://drive.google.com/file/d/1rCiSdLzaN-uZbIM-tWs4RuzZB2aePaYt/view?usp=sharing

** The Dataset being generated after executing these benchmarks in several experiments, enabling further analysis which has been used for our GPU performance model's implementation and Machine Learning-based discoveries can be found at: https://drive.google.com/file/d/1weuTCvvWf74GLJMMn8Bw8Vb-Yw69SJcm/view?usp=sharing **

## C# based Performance Model Structure
_Location_: The `CSharp_MSc_Final_Project` folder contains a Windows Forms application (C#, .NET Framework 4.5.2) built on top of the CUDA benchmark results. Open `CSharp_MSc_Final_Project/MSc_Final_Project.sln` in Visual Studio; the application starts from the `CoScheduler` form.

```
CSharp_MSc_Final_Project/
├── MSc_Final_Project.sln                      Visual Studio solution
└── MSc_Final_Project/
    ├── MSc_Final_Project.csproj               Project file
    ├── Program.cs                             Entry point (opens CoScheduler)
    ├── CoScheduler.cs                         Main form
    ├── ProgramSet.cs, StreamSetPrediction.cs,
    │   MemCpyPrediction.cs, SchedulerPrediction.cs,
    │   ActualExecutions.cs, Combination_Panel.cs   Other forms (each with .Designer.cs and .resx)
    ├── Linq.dbml, Linq.designer.cs            LINQ to SQL model of the database
    ├── SchedulerDataBase.mdf, _log.ldf        SQL Server LocalDB database
    └── Properties/                            Assembly info, resources and settings
```

_Prerequisites_ (Windows only):
- Visual Studio 2015 or newer, with the .NET Framework 4.5.2 targeting pack
- SQL Server Express LocalDB (`MSSQLLocalDB` instance), included in Visual Studio's "Data storage and processing" workload
- Microsoft Office with Excel: the project references `Microsoft.Office.Interop.Excel`, so it does not build without it
- Microsoft Access Database Engine (`Microsoft.ACE.OLEDB.12.0`), 32-bit (x86) version: used to read the `.xls`/`.xlsx` input files
- ExcelDataReader 3.6.0: restored automatically by NuGet from `packages.config` on the first build

_Build & Run_: Select the **x86** platform in Visual Studio, not **Any CPU**. The application looks for `SchedulerDataBase.mdf` three folders above its output folder (`bin\x86\Debug\` → project folder); with Any CPU the output goes to `bin\Debug\` and the database is not found.

_Database Notes_:
- `SchedulerDataBase.mdf` and `SchedulerDataBase_log.ldf` hold the tables defined in `Linq.dbml` and must stay together in the project folder.
- Running the application writes to these files, so Git will show them as modified.
- A newer LocalDB version may upgrade the `.mdf` the first time it is opened; after that, older LocalDB versions cannot open it.
