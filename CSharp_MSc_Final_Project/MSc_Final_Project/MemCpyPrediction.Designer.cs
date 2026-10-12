namespace MSc_Final_Project
{
    partial class MemCpyPrediction
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
            this.MemCpy_GridView = new System.Windows.Forms.DataGridView();
            this.SelectActualMemoryCpyFile_Btn = new System.Windows.Forms.Button();
            this.LoadActualMemoryCpyFile_Btn = new System.Windows.Forms.Button();
            this.SelectCpyFile_TxtBox = new System.Windows.Forms.TextBox();
            this.Actual_MrmcpyFile_Worksheet_Lbl = new System.Windows.Forms.Label();
            this.RemoveNonCpyRows_Btn = new System.Windows.Forms.Button();
            this.CpyAvgError_Btn = new System.Windows.Forms.Button();
            this.HtoDAvg_TxtBox = new System.Windows.Forms.TextBox();
            this.DtoHAvg_TxtBox = new System.Windows.Forms.TextBox();
            this.HtoDAvg_Lbl = new System.Windows.Forms.Label();
            this.DtoHAvg_Lbl = new System.Windows.Forms.Label();
            this.CpyFile_openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.CpyFileWorksheets_ComboBox = new System.Windows.Forms.ComboBox();
            this.ActualDtoHAvg_TxtBox = new System.Windows.Forms.TextBox();
            this.ActualHtoDAvg_TxtBox = new System.Windows.Forms.TextBox();
            this.ActualHtoD_Lbl = new System.Windows.Forms.Label();
            this.ActualDtoH_Lbl = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.MemCpy_GridView)).BeginInit();
            this.SuspendLayout();
            // 
            // MemCpy_GridView
            // 
            this.MemCpy_GridView.AllowUserToAddRows = false;
            this.MemCpy_GridView.AllowUserToDeleteRows = false;
            this.MemCpy_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.MemCpy_GridView.Location = new System.Drawing.Point(14, 166);
            this.MemCpy_GridView.MaximumSize = new System.Drawing.Size(353, 375);
            this.MemCpy_GridView.MinimumSize = new System.Drawing.Size(353, 375);
            this.MemCpy_GridView.Name = "MemCpy_GridView";
            this.MemCpy_GridView.ReadOnly = true;
            this.MemCpy_GridView.RowHeadersVisible = false;
            this.MemCpy_GridView.Size = new System.Drawing.Size(353, 375);
            this.MemCpy_GridView.TabIndex = 0;
            // 
            // SelectActualMemoryCpyFile_Btn
            // 
            this.SelectActualMemoryCpyFile_Btn.Location = new System.Drawing.Point(14, 29);
            this.SelectActualMemoryCpyFile_Btn.Name = "SelectActualMemoryCpyFile_Btn";
            this.SelectActualMemoryCpyFile_Btn.Size = new System.Drawing.Size(177, 25);
            this.SelectActualMemoryCpyFile_Btn.TabIndex = 1;
            this.SelectActualMemoryCpyFile_Btn.Text = "Select  Actual Memory Copy File";
            this.SelectActualMemoryCpyFile_Btn.UseVisualStyleBackColor = true;
            this.SelectActualMemoryCpyFile_Btn.Click += new System.EventHandler(this.SelectActualMemoryCpyFile_Btn_Click);
            // 
            // LoadActualMemoryCpyFile_Btn
            // 
            this.LoadActualMemoryCpyFile_Btn.Location = new System.Drawing.Point(160, 125);
            this.LoadActualMemoryCpyFile_Btn.Name = "LoadActualMemoryCpyFile_Btn";
            this.LoadActualMemoryCpyFile_Btn.Size = new System.Drawing.Size(84, 25);
            this.LoadActualMemoryCpyFile_Btn.TabIndex = 2;
            this.LoadActualMemoryCpyFile_Btn.Text = "Load";
            this.LoadActualMemoryCpyFile_Btn.UseVisualStyleBackColor = true;
            this.LoadActualMemoryCpyFile_Btn.Click += new System.EventHandler(this.LoadActualMemoryCpyFile_Btn_Click);
            // 
            // SelectCpyFile_TxtBox
            // 
            this.SelectCpyFile_TxtBox.Location = new System.Drawing.Point(210, 31);
            this.SelectCpyFile_TxtBox.Name = "SelectCpyFile_TxtBox";
            this.SelectCpyFile_TxtBox.ReadOnly = true;
            this.SelectCpyFile_TxtBox.Size = new System.Drawing.Size(157, 21);
            this.SelectCpyFile_TxtBox.TabIndex = 3;
            // 
            // Actual_MrmcpyFile_Worksheet_Lbl
            // 
            this.Actual_MrmcpyFile_Worksheet_Lbl.AutoSize = true;
            this.Actual_MrmcpyFile_Worksheet_Lbl.Location = new System.Drawing.Point(129, 85);
            this.Actual_MrmcpyFile_Worksheet_Lbl.Name = "Actual_MrmcpyFile_Worksheet_Lbl";
            this.Actual_MrmcpyFile_Worksheet_Lbl.Size = new System.Drawing.Size(62, 14);
            this.Actual_MrmcpyFile_Worksheet_Lbl.TabIndex = 5;
            this.Actual_MrmcpyFile_Worksheet_Lbl.Text = "Worksheet:";
            // 
            // RemoveNonCpyRows_Btn
            // 
            this.RemoveNonCpyRows_Btn.Location = new System.Drawing.Point(33, 558);
            this.RemoveNonCpyRows_Btn.Name = "RemoveNonCpyRows_Btn";
            this.RemoveNonCpyRows_Btn.Size = new System.Drawing.Size(87, 25);
            this.RemoveNonCpyRows_Btn.TabIndex = 6;
            this.RemoveNonCpyRows_Btn.Text = "Remove non-Copy Rows";
            this.RemoveNonCpyRows_Btn.UseVisualStyleBackColor = true;
            this.RemoveNonCpyRows_Btn.Click += new System.EventHandler(this.RemoveNonCpyRows_Btn_Click);
            // 
            // CpyAvgError_Btn
            // 
            this.CpyAvgError_Btn.Location = new System.Drawing.Point(14, 619);
            this.CpyAvgError_Btn.Name = "CpyAvgError_Btn";
            this.CpyAvgError_Btn.Size = new System.Drawing.Size(126, 38);
            this.CpyAvgError_Btn.TabIndex = 7;
            this.CpyAvgError_Btn.Text = "Calculate HtoD and DtoH Avg Error";
            this.CpyAvgError_Btn.UseVisualStyleBackColor = true;
            this.CpyAvgError_Btn.Click += new System.EventHandler(this.CpyAvgError_Btn_Click);
            // 
            // HtoDAvg_TxtBox
            // 
            this.HtoDAvg_TxtBox.Location = new System.Drawing.Point(270, 609);
            this.HtoDAvg_TxtBox.Name = "HtoDAvg_TxtBox";
            this.HtoDAvg_TxtBox.ReadOnly = true;
            this.HtoDAvg_TxtBox.Size = new System.Drawing.Size(97, 21);
            this.HtoDAvg_TxtBox.TabIndex = 8;
            // 
            // DtoHAvg_TxtBox
            // 
            this.DtoHAvg_TxtBox.Location = new System.Drawing.Point(270, 636);
            this.DtoHAvg_TxtBox.Name = "DtoHAvg_TxtBox";
            this.DtoHAvg_TxtBox.ReadOnly = true;
            this.DtoHAvg_TxtBox.Size = new System.Drawing.Size(97, 21);
            this.DtoHAvg_TxtBox.TabIndex = 9;
            // 
            // HtoDAvg_Lbl
            // 
            this.HtoDAvg_Lbl.AutoSize = true;
            this.HtoDAvg_Lbl.Location = new System.Drawing.Point(157, 612);
            this.HtoDAvg_Lbl.Name = "HtoDAvg_Lbl";
            this.HtoDAvg_Lbl.Size = new System.Drawing.Size(106, 14);
            this.HtoDAvg_Lbl.TabIndex = 10;
            this.HtoDAvg_Lbl.Text = "HtoD Avg Error (%):";
            // 
            // DtoHAvg_Lbl
            // 
            this.DtoHAvg_Lbl.AutoSize = true;
            this.DtoHAvg_Lbl.Location = new System.Drawing.Point(158, 639);
            this.DtoHAvg_Lbl.Name = "DtoHAvg_Lbl";
            this.DtoHAvg_Lbl.Size = new System.Drawing.Size(106, 14);
            this.DtoHAvg_Lbl.TabIndex = 11;
            this.DtoHAvg_Lbl.Text = "DtoH Avg Error (%):";
            // 
            // CpyFile_openFileDialog
            // 
            this.CpyFile_openFileDialog.FileName = "CpyFile_openFileDialog";
            // 
            // CpyFileWorksheets_ComboBox
            // 
            this.CpyFileWorksheets_ComboBox.FormattingEnabled = true;
            this.CpyFileWorksheets_ComboBox.Location = new System.Drawing.Point(210, 82);
            this.CpyFileWorksheets_ComboBox.Name = "CpyFileWorksheets_ComboBox";
            this.CpyFileWorksheets_ComboBox.Size = new System.Drawing.Size(121, 22);
            this.CpyFileWorksheets_ComboBox.TabIndex = 12;
            this.CpyFileWorksheets_ComboBox.SelectedIndexChanged += new System.EventHandler(this.CpyFileWorksheets_ComboBox_SelectedIndexChanged);
            // 
            // ActualDtoHAvg_TxtBox
            // 
            this.ActualDtoHAvg_TxtBox.Location = new System.Drawing.Point(270, 582);
            this.ActualDtoHAvg_TxtBox.Name = "ActualDtoHAvg_TxtBox";
            this.ActualDtoHAvg_TxtBox.ReadOnly = true;
            this.ActualDtoHAvg_TxtBox.Size = new System.Drawing.Size(97, 21);
            this.ActualDtoHAvg_TxtBox.TabIndex = 13;
            // 
            // ActualHtoDAvg_TxtBox
            // 
            this.ActualHtoDAvg_TxtBox.Location = new System.Drawing.Point(270, 555);
            this.ActualHtoDAvg_TxtBox.Name = "ActualHtoDAvg_TxtBox";
            this.ActualHtoDAvg_TxtBox.ReadOnly = true;
            this.ActualHtoDAvg_TxtBox.Size = new System.Drawing.Size(97, 21);
            this.ActualHtoDAvg_TxtBox.TabIndex = 14;
            // 
            // ActualHtoD_Lbl
            // 
            this.ActualHtoD_Lbl.AutoSize = true;
            this.ActualHtoD_Lbl.Location = new System.Drawing.Point(157, 558);
            this.ActualHtoD_Lbl.Name = "ActualHtoD_Lbl";
            this.ActualHtoD_Lbl.Size = new System.Drawing.Size(107, 14);
            this.ActualHtoD_Lbl.TabIndex = 15;
            this.ActualHtoD_Lbl.Text = "Actual HtoD in Avg:";
            // 
            // ActualDtoH_Lbl
            // 
            this.ActualDtoH_Lbl.AutoSize = true;
            this.ActualDtoH_Lbl.Location = new System.Drawing.Point(157, 585);
            this.ActualDtoH_Lbl.Name = "ActualDtoH_Lbl";
            this.ActualDtoH_Lbl.Size = new System.Drawing.Size(107, 14);
            this.ActualDtoH_Lbl.TabIndex = 16;
            this.ActualDtoH_Lbl.Text = "Actual DtoH in Avg:";
            // 
            // MemCpyPrediction
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 14F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(381, 671);
            this.Controls.Add(this.ActualDtoH_Lbl);
            this.Controls.Add(this.ActualHtoD_Lbl);
            this.Controls.Add(this.ActualHtoDAvg_TxtBox);
            this.Controls.Add(this.ActualDtoHAvg_TxtBox);
            this.Controls.Add(this.CpyFileWorksheets_ComboBox);
            this.Controls.Add(this.DtoHAvg_Lbl);
            this.Controls.Add(this.HtoDAvg_Lbl);
            this.Controls.Add(this.DtoHAvg_TxtBox);
            this.Controls.Add(this.HtoDAvg_TxtBox);
            this.Controls.Add(this.CpyAvgError_Btn);
            this.Controls.Add(this.RemoveNonCpyRows_Btn);
            this.Controls.Add(this.Actual_MrmcpyFile_Worksheet_Lbl);
            this.Controls.Add(this.SelectCpyFile_TxtBox);
            this.Controls.Add(this.LoadActualMemoryCpyFile_Btn);
            this.Controls.Add(this.SelectActualMemoryCpyFile_Btn);
            this.Controls.Add(this.MemCpy_GridView);
            this.Font = new System.Drawing.Font("Linux Biolinum", 9F);
            this.MaximizeBox = false;
            this.Name = "MemCpyPrediction";
            this.Text = "Predict Memory Copies";
            ((System.ComponentModel.ISupportInitialize)(this.MemCpy_GridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView MemCpy_GridView;
        private System.Windows.Forms.Button SelectActualMemoryCpyFile_Btn;
        private System.Windows.Forms.Button LoadActualMemoryCpyFile_Btn;
        private System.Windows.Forms.TextBox SelectCpyFile_TxtBox;
        private System.Windows.Forms.Label Actual_MrmcpyFile_Worksheet_Lbl;
        private System.Windows.Forms.Button RemoveNonCpyRows_Btn;
        private System.Windows.Forms.Button CpyAvgError_Btn;
        private System.Windows.Forms.TextBox DtoHAvg_TxtBox;
        private System.Windows.Forms.Label HtoDAvg_Lbl;
        private System.Windows.Forms.Label DtoHAvg_Lbl;
        private System.Windows.Forms.TextBox HtoDAvg_TxtBox;
        private System.Windows.Forms.OpenFileDialog CpyFile_openFileDialog;
        private System.Windows.Forms.ComboBox CpyFileWorksheets_ComboBox;
        private System.Windows.Forms.TextBox ActualDtoHAvg_TxtBox;
        private System.Windows.Forms.TextBox ActualHtoDAvg_TxtBox;
        private System.Windows.Forms.Label ActualHtoD_Lbl;
        private System.Windows.Forms.Label ActualDtoH_Lbl;
    }
}