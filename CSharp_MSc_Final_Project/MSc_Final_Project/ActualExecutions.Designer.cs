namespace MSc_Final_Project
{
    partial class ActualExecutions
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.AEFile_GridView = new System.Windows.Forms.DataGridView();
            this.backgroundWorker1 = new System.ComponentModel.BackgroundWorker();
            this.AE_GridView = new System.Windows.Forms.DataGridView();
            this.StreamSet = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Duration = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.AEFile_Btn = new System.Windows.Forms.Button();
            this.AEFile_TxtBox = new System.Windows.Forms.TextBox();
            this.AEFileWorkSheets_Lbl = new System.Windows.Forms.Label();
            this.AEWorkSheet_ComboBox = new System.Windows.Forms.ComboBox();
            this.AESort_Btn = new System.Windows.Forms.Button();
            this.ExportSortedAE_Btn = new System.Windows.Forms.Button();
            this.AEopenFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.AEFileLoad_Btn = new System.Windows.Forms.Button();
            this.AEsaveFileDialog = new System.Windows.Forms.SaveFileDialog();
            ((System.ComponentModel.ISupportInitialize)(this.AEFile_GridView)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.AE_GridView)).BeginInit();
            this.SuspendLayout();
            // 
            // AEFile_GridView
            // 
            this.AEFile_GridView.AllowUserToAddRows = false;
            this.AEFile_GridView.AllowUserToDeleteRows = false;
            this.AEFile_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.AEFile_GridView.Location = new System.Drawing.Point(12, 182);
            this.AEFile_GridView.Name = "AEFile_GridView";
            this.AEFile_GridView.ReadOnly = true;
            this.AEFile_GridView.RowHeadersVisible = false;
            this.AEFile_GridView.Size = new System.Drawing.Size(360, 348);
            this.AEFile_GridView.TabIndex = 0;
            // 
            // AE_GridView
            // 
            this.AE_GridView.AllowUserToAddRows = false;
            this.AE_GridView.AllowUserToDeleteRows = false;
            this.AE_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.AE_GridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.StreamSet,
            this.Duration});
            this.AE_GridView.Location = new System.Drawing.Point(389, 182);
            this.AE_GridView.Name = "AE_GridView";
            this.AE_GridView.ReadOnly = true;
            this.AE_GridView.RowHeadersVisible = false;
            this.AE_GridView.Size = new System.Drawing.Size(303, 348);
            this.AE_GridView.TabIndex = 1;
            // 
            // StreamSet
            // 
            this.StreamSet.HeaderText = "Stream Set";
            this.StreamSet.MaxInputLength = 327670;
            this.StreamSet.Name = "StreamSet";
            this.StreamSet.ReadOnly = true;
            this.StreamSet.Width = 150;
            // 
            // Duration
            // 
            this.Duration.HeaderText = "Duration(ns)";
            this.Duration.Name = "Duration";
            this.Duration.ReadOnly = true;
            this.Duration.Width = 150;
            // 
            // AEFile_Btn
            // 
            this.AEFile_Btn.Location = new System.Drawing.Point(14, 31);
            this.AEFile_Btn.Name = "AEFile_Btn";
            this.AEFile_Btn.Size = new System.Drawing.Size(120, 36);
            this.AEFile_Btn.TabIndex = 2;
            this.AEFile_Btn.Text = "Actual Execution File: ";
            this.AEFile_Btn.UseVisualStyleBackColor = true;
            this.AEFile_Btn.Click += new System.EventHandler(this.AEFile_Btn_Click);
            // 
            // AEFile_TxtBox
            // 
            this.AEFile_TxtBox.Location = new System.Drawing.Point(140, 39);
            this.AEFile_TxtBox.Name = "AEFile_TxtBox";
            this.AEFile_TxtBox.ReadOnly = true;
            this.AEFile_TxtBox.Size = new System.Drawing.Size(232, 21);
            this.AEFile_TxtBox.TabIndex = 3;
            // 
            // AEFileWorkSheets_Lbl
            // 
            this.AEFileWorkSheets_Lbl.AutoSize = true;
            this.AEFileWorkSheets_Lbl.Font = new System.Drawing.Font("Linux Biolinum", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.AEFileWorkSheets_Lbl.Location = new System.Drawing.Point(62, 92);
            this.AEFileWorkSheets_Lbl.Name = "AEFileWorkSheets_Lbl";
            this.AEFileWorkSheets_Lbl.Size = new System.Drawing.Size(72, 15);
            this.AEFileWorkSheets_Lbl.TabIndex = 4;
            this.AEFileWorkSheets_Lbl.Text = "WorkSheet: ";
            // 
            // AEWorkSheet_ComboBox
            // 
            this.AEWorkSheet_ComboBox.FormattingEnabled = true;
            this.AEWorkSheet_ComboBox.Location = new System.Drawing.Point(140, 90);
            this.AEWorkSheet_ComboBox.Name = "AEWorkSheet_ComboBox";
            this.AEWorkSheet_ComboBox.Size = new System.Drawing.Size(121, 22);
            this.AEWorkSheet_ComboBox.TabIndex = 5;
            this.AEWorkSheet_ComboBox.SelectedIndexChanged += new System.EventHandler(this.AEWorkSheet_ComboBox_SelectedIndexChanged);
            // 
            // AESort_Btn
            // 
            this.AESort_Btn.Location = new System.Drawing.Point(476, 39);
            this.AESort_Btn.Name = "AESort_Btn";
            this.AESort_Btn.Size = new System.Drawing.Size(117, 46);
            this.AESort_Btn.TabIndex = 6;
            this.AESort_Btn.Text = "Sort";
            this.AESort_Btn.UseVisualStyleBackColor = true;
            this.AESort_Btn.Click += new System.EventHandler(this.AESort_Btn_Click);
            // 
            // ExportSortedAE_Btn
            // 
            this.ExportSortedAE_Btn.Location = new System.Drawing.Point(476, 102);
            this.ExportSortedAE_Btn.Name = "ExportSortedAE_Btn";
            this.ExportSortedAE_Btn.Size = new System.Drawing.Size(117, 46);
            this.ExportSortedAE_Btn.TabIndex = 7;
            this.ExportSortedAE_Btn.Text = "Export to Excel File";
            this.ExportSortedAE_Btn.UseVisualStyleBackColor = true;
            this.ExportSortedAE_Btn.Click += new System.EventHandler(this.ExportSortedAE_Btn_Click);
            // 
            // AEopenFileDialog
            // 
            this.AEopenFileDialog.FileName = "openFileDialog1";
            // 
            // AEFileLoad_Btn
            // 
            this.AEFileLoad_Btn.Location = new System.Drawing.Point(134, 138);
            this.AEFileLoad_Btn.Name = "AEFileLoad_Btn";
            this.AEFileLoad_Btn.Size = new System.Drawing.Size(127, 38);
            this.AEFileLoad_Btn.TabIndex = 8;
            this.AEFileLoad_Btn.Text = "Load";
            this.AEFileLoad_Btn.UseVisualStyleBackColor = true;
            this.AEFileLoad_Btn.Click += new System.EventHandler(this.AEFileLoad_Btn_Click);
            // 
            // ActualExecutions
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 14F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(704, 542);
            this.Controls.Add(this.AEFileLoad_Btn);
            this.Controls.Add(this.ExportSortedAE_Btn);
            this.Controls.Add(this.AESort_Btn);
            this.Controls.Add(this.AEWorkSheet_ComboBox);
            this.Controls.Add(this.AEFileWorkSheets_Lbl);
            this.Controls.Add(this.AEFile_TxtBox);
            this.Controls.Add(this.AEFile_Btn);
            this.Controls.Add(this.AE_GridView);
            this.Controls.Add(this.AEFile_GridView);
            this.Font = new System.Drawing.Font("Linux Biolinum", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(720, 581);
            this.MinimumSize = new System.Drawing.Size(720, 581);
            this.Name = "ActualExecutions";
            this.Text = "ActualExecutions";
            this.Load += new System.EventHandler(this.ActualExecutions_Load);
            ((System.ComponentModel.ISupportInitialize)(this.AEFile_GridView)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.AE_GridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView AEFile_GridView;
        private System.ComponentModel.BackgroundWorker backgroundWorker1;
        private System.Windows.Forms.DataGridView AE_GridView;
        private System.Windows.Forms.Button AEFile_Btn;
        private System.Windows.Forms.TextBox AEFile_TxtBox;
        private System.Windows.Forms.Label AEFileWorkSheets_Lbl;
        private System.Windows.Forms.ComboBox AEWorkSheet_ComboBox;
        private System.Windows.Forms.Button AESort_Btn;
        private System.Windows.Forms.Button ExportSortedAE_Btn;
        private System.Windows.Forms.OpenFileDialog AEopenFileDialog;
        private System.Windows.Forms.Button AEFileLoad_Btn;
        private System.Windows.Forms.DataGridViewTextBoxColumn StreamSet;
        private System.Windows.Forms.DataGridViewTextBoxColumn Duration;
        private System.Windows.Forms.SaveFileDialog AEsaveFileDialog;
    }
}