namespace MSc_Final_Project
{
    partial class StreamSetPrediction
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
            this.PredictStreamSets_Btn = new System.Windows.Forms.Button();
            this.PredictedStreamSets_GridView = new System.Windows.Forms.DataGridView();
            this.StreamSet = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Duration = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ASFile_GridView = new System.Windows.Forms.DataGridView();
            this.StreamSetsError_GridView = new System.Windows.Forms.DataGridView();
            this.Error = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SelectASFile_Btn = new System.Windows.Forms.Button();
            this.StreamSetsError_Btn = new System.Windows.Forms.Button();
            this.ASFileLocation_TxtBot = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.ASFileWorksheet_ComboBox = new System.Windows.Forms.ComboBox();
            this.LoadASFile_Btn = new System.Windows.Forms.Button();
            this.ASopenFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.PredictionProgress_Lbl = new System.Windows.Forms.Label();
            this.PredictionResults_Btn = new System.Windows.Forms.Button();
            this.ExportStreamSet_Btn = new System.Windows.Forms.Button();
            this.StreamSetsaveFileDialog = new System.Windows.Forms.SaveFileDialog();
            this.nonStream_GridView = new System.Windows.Forms.DataGridView();
            this.StreamSets = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Durations = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.nonStreamBtn = new System.Windows.Forms.Button();
            this.ExportnonStreamBtn = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.PredictedStreamSets_GridView)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ASFile_GridView)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.StreamSetsError_GridView)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.nonStream_GridView)).BeginInit();
            this.SuspendLayout();
            // 
            // PredictStreamSets_Btn
            // 
            this.PredictStreamSets_Btn.Location = new System.Drawing.Point(251, 12);
            this.PredictStreamSets_Btn.Name = "PredictStreamSets_Btn";
            this.PredictStreamSets_Btn.Size = new System.Drawing.Size(117, 48);
            this.PredictStreamSets_Btn.TabIndex = 0;
            this.PredictStreamSets_Btn.Text = "Predict Stream Sets";
            this.PredictStreamSets_Btn.UseVisualStyleBackColor = true;
            this.PredictStreamSets_Btn.Click += new System.EventHandler(this.PredictStreamSets_Btn_Click);
            // 
            // PredictedStreamSets_GridView
            // 
            this.PredictedStreamSets_GridView.AllowUserToAddRows = false;
            this.PredictedStreamSets_GridView.AllowUserToDeleteRows = false;
            this.PredictedStreamSets_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.PredictedStreamSets_GridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.StreamSet,
            this.Duration});
            this.PredictedStreamSets_GridView.Location = new System.Drawing.Point(248, 142);
            this.PredictedStreamSets_GridView.Name = "PredictedStreamSets_GridView";
            this.PredictedStreamSets_GridView.ReadOnly = true;
            this.PredictedStreamSets_GridView.RowHeadersVisible = false;
            this.PredictedStreamSets_GridView.Size = new System.Drawing.Size(221, 327);
            this.PredictedStreamSets_GridView.TabIndex = 1;
            // 
            // StreamSet
            // 
            this.StreamSet.HeaderText = "Stream Set";
            this.StreamSet.Name = "StreamSet";
            this.StreamSet.ReadOnly = true;
            this.StreamSet.Width = 116;
            // 
            // Duration
            // 
            this.Duration.HeaderText = "Duration(ns)";
            this.Duration.Name = "Duration";
            this.Duration.ReadOnly = true;
            // 
            // ASFile_GridView
            // 
            this.ASFile_GridView.AllowUserToAddRows = false;
            this.ASFile_GridView.AllowUserToDeleteRows = false;
            this.ASFile_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.ASFile_GridView.Location = new System.Drawing.Point(482, 142);
            this.ASFile_GridView.Name = "ASFile_GridView";
            this.ASFile_GridView.ReadOnly = true;
            this.ASFile_GridView.RowHeadersVisible = false;
            this.ASFile_GridView.Size = new System.Drawing.Size(221, 327);
            this.ASFile_GridView.TabIndex = 2;
            // 
            // StreamSetsError_GridView
            // 
            this.StreamSetsError_GridView.AllowUserToAddRows = false;
            this.StreamSetsError_GridView.AllowUserToDeleteRows = false;
            this.StreamSetsError_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.StreamSetsError_GridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.Error});
            this.StreamSetsError_GridView.Location = new System.Drawing.Point(718, 142);
            this.StreamSetsError_GridView.Name = "StreamSetsError_GridView";
            this.StreamSetsError_GridView.ReadOnly = true;
            this.StreamSetsError_GridView.RowHeadersVisible = false;
            this.StreamSetsError_GridView.Size = new System.Drawing.Size(104, 327);
            this.StreamSetsError_GridView.TabIndex = 3;
            // 
            // Error
            // 
            this.Error.HeaderText = "Error(%)";
            this.Error.Name = "Error";
            this.Error.ReadOnly = true;
            // 
            // SelectASFile_Btn
            // 
            this.SelectASFile_Btn.Location = new System.Drawing.Point(482, 12);
            this.SelectASFile_Btn.Name = "SelectASFile_Btn";
            this.SelectASFile_Btn.Size = new System.Drawing.Size(87, 42);
            this.SelectASFile_Btn.TabIndex = 4;
            this.SelectASFile_Btn.Text = "Actual StreamSets";
            this.SelectASFile_Btn.UseVisualStyleBackColor = true;
            this.SelectASFile_Btn.Click += new System.EventHandler(this.SelectASFile_Btn_Click);
            // 
            // StreamSetsError_Btn
            // 
            this.StreamSetsError_Btn.Location = new System.Drawing.Point(732, 55);
            this.StreamSetsError_Btn.Name = "StreamSetsError_Btn";
            this.StreamSetsError_Btn.Size = new System.Drawing.Size(75, 48);
            this.StreamSetsError_Btn.TabIndex = 5;
            this.StreamSetsError_Btn.Text = "Model Error";
            this.StreamSetsError_Btn.UseVisualStyleBackColor = true;
            this.StreamSetsError_Btn.Click += new System.EventHandler(this.StreamSetsError_Btn_Click);
            // 
            // ASFileLocation_TxtBot
            // 
            this.ASFileLocation_TxtBot.Location = new System.Drawing.Point(575, 23);
            this.ASFileLocation_TxtBot.Name = "ASFileLocation_TxtBot";
            this.ASFileLocation_TxtBot.ReadOnly = true;
            this.ASFileLocation_TxtBot.Size = new System.Drawing.Size(128, 21);
            this.ASFileLocation_TxtBot.TabIndex = 6;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(504, 72);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(65, 14);
            this.label1.TabIndex = 7;
            this.label1.Text = "Worksheet: ";
            // 
            // ASFileWorksheet_ComboBox
            // 
            this.ASFileWorksheet_ComboBox.FormattingEnabled = true;
            this.ASFileWorksheet_ComboBox.Location = new System.Drawing.Point(575, 69);
            this.ASFileWorksheet_ComboBox.Name = "ASFileWorksheet_ComboBox";
            this.ASFileWorksheet_ComboBox.Size = new System.Drawing.Size(128, 22);
            this.ASFileWorksheet_ComboBox.TabIndex = 8;
            this.ASFileWorksheet_ComboBox.SelectedIndexChanged += new System.EventHandler(this.ASFileWorksheet_ComboBox_SelectedIndexChanged);
            // 
            // LoadASFile_Btn
            // 
            this.LoadASFile_Btn.Location = new System.Drawing.Point(556, 104);
            this.LoadASFile_Btn.Name = "LoadASFile_Btn";
            this.LoadASFile_Btn.Size = new System.Drawing.Size(85, 32);
            this.LoadASFile_Btn.TabIndex = 9;
            this.LoadASFile_Btn.Text = "Load";
            this.LoadASFile_Btn.UseVisualStyleBackColor = true;
            this.LoadASFile_Btn.Click += new System.EventHandler(this.LoadASFile_Btn_Click);
            // 
            // ASopenFileDialog
            // 
            this.ASopenFileDialog.FileName = "ASopenFileDialog";
            // 
            // PredictionProgress_Lbl
            // 
            this.PredictionProgress_Lbl.AutoSize = true;
            this.PredictionProgress_Lbl.Location = new System.Drawing.Point(257, 72);
            this.PredictionProgress_Lbl.Name = "PredictionProgress_Lbl";
            this.PredictionProgress_Lbl.Size = new System.Drawing.Size(31, 14);
            this.PredictionProgress_Lbl.TabIndex = 11;
            this.PredictionProgress_Lbl.Text = "Idle..";
            // 
            // PredictionResults_Btn
            // 
            this.PredictionResults_Btn.Location = new System.Drawing.Point(374, 12);
            this.PredictionResults_Btn.Name = "PredictionResults_Btn";
            this.PredictionResults_Btn.Size = new System.Drawing.Size(81, 48);
            this.PredictionResults_Btn.TabIndex = 12;
            this.PredictionResults_Btn.Text = "Load Results";
            this.PredictionResults_Btn.UseVisualStyleBackColor = true;
            this.PredictionResults_Btn.Click += new System.EventHandler(this.PredictionResults_Btn_Click);
            // 
            // ExportStreamSet_Btn
            // 
            this.ExportStreamSet_Btn.Location = new System.Drawing.Point(305, 104);
            this.ExportStreamSet_Btn.Name = "ExportStreamSet_Btn";
            this.ExportStreamSet_Btn.Size = new System.Drawing.Size(108, 32);
            this.ExportStreamSet_Btn.TabIndex = 13;
            this.ExportStreamSet_Btn.Text = "Export to Excel File";
            this.ExportStreamSet_Btn.UseVisualStyleBackColor = true;
            this.ExportStreamSet_Btn.Click += new System.EventHandler(this.ExportStreamSet_Btn_Click);
            // 
            // nonStream_GridView
            // 
            this.nonStream_GridView.AllowUserToAddRows = false;
            this.nonStream_GridView.AllowUserToDeleteRows = false;
            this.nonStream_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.nonStream_GridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.StreamSets,
            this.Durations});
            this.nonStream_GridView.Location = new System.Drawing.Point(12, 142);
            this.nonStream_GridView.Name = "nonStream_GridView";
            this.nonStream_GridView.ReadOnly = true;
            this.nonStream_GridView.RowHeadersVisible = false;
            this.nonStream_GridView.Size = new System.Drawing.Size(221, 327);
            this.nonStream_GridView.TabIndex = 14;
            // 
            // StreamSets
            // 
            this.StreamSets.HeaderText = "Stream Set";
            this.StreamSets.Name = "StreamSets";
            this.StreamSets.ReadOnly = true;
            this.StreamSets.Width = 116;
            // 
            // Durations
            // 
            this.Durations.HeaderText = "Duration(ns)";
            this.Durations.Name = "Durations";
            this.Durations.ReadOnly = true;
            // 
            // nonStreamBtn
            // 
            this.nonStreamBtn.Location = new System.Drawing.Point(65, 12);
            this.nonStreamBtn.Name = "nonStreamBtn";
            this.nonStreamBtn.Size = new System.Drawing.Size(126, 48);
            this.nonStreamBtn.TabIndex = 15;
            this.nonStreamBtn.Text = "Load nonStream Sets";
            this.nonStreamBtn.UseVisualStyleBackColor = true;
            this.nonStreamBtn.Click += new System.EventHandler(this.nonStreamBtn_Click);
            // 
            // ExportnonStreamBtn
            // 
            this.ExportnonStreamBtn.Location = new System.Drawing.Point(65, 88);
            this.ExportnonStreamBtn.Name = "ExportnonStreamBtn";
            this.ExportnonStreamBtn.Size = new System.Drawing.Size(126, 48);
            this.ExportnonStreamBtn.TabIndex = 16;
            this.ExportnonStreamBtn.Text = "Export nonStream Sets";
            this.ExportnonStreamBtn.UseVisualStyleBackColor = true;
            this.ExportnonStreamBtn.Click += new System.EventHandler(this.ExportnonStreamBtn_Click);
            // 
            // StreamSetPrediction
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 14F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(838, 481);
            this.Controls.Add(this.ExportnonStreamBtn);
            this.Controls.Add(this.nonStreamBtn);
            this.Controls.Add(this.nonStream_GridView);
            this.Controls.Add(this.ExportStreamSet_Btn);
            this.Controls.Add(this.PredictionResults_Btn);
            this.Controls.Add(this.PredictionProgress_Lbl);
            this.Controls.Add(this.LoadASFile_Btn);
            this.Controls.Add(this.ASFileWorksheet_ComboBox);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.ASFileLocation_TxtBot);
            this.Controls.Add(this.StreamSetsError_Btn);
            this.Controls.Add(this.SelectASFile_Btn);
            this.Controls.Add(this.StreamSetsError_GridView);
            this.Controls.Add(this.ASFile_GridView);
            this.Controls.Add(this.PredictedStreamSets_GridView);
            this.Controls.Add(this.PredictStreamSets_Btn);
            this.Font = new System.Drawing.Font("Linux Biolinum", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(854, 520);
            this.MinimumSize = new System.Drawing.Size(854, 520);
            this.Name = "StreamSetPrediction";
            this.Text = "StreamPair Prediction";
            this.Load += new System.EventHandler(this.StreamSetPrediction_Load);
            ((System.ComponentModel.ISupportInitialize)(this.PredictedStreamSets_GridView)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ASFile_GridView)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.StreamSetsError_GridView)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.nonStream_GridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button PredictStreamSets_Btn;
        private System.Windows.Forms.DataGridView PredictedStreamSets_GridView;
        private System.Windows.Forms.DataGridView ASFile_GridView;
        private System.Windows.Forms.DataGridView StreamSetsError_GridView;
        private System.Windows.Forms.Button SelectASFile_Btn;
        private System.Windows.Forms.Button StreamSetsError_Btn;
        private System.Windows.Forms.TextBox ASFileLocation_TxtBot;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox ASFileWorksheet_ComboBox;
        private System.Windows.Forms.Button LoadASFile_Btn;
        private System.Windows.Forms.OpenFileDialog ASopenFileDialog;
        private System.Windows.Forms.DataGridViewTextBoxColumn StreamSet;
        private System.Windows.Forms.DataGridViewTextBoxColumn Duration;
        private System.Windows.Forms.Label PredictionProgress_Lbl;
        private System.Windows.Forms.Button PredictionResults_Btn;
        private System.Windows.Forms.DataGridViewTextBoxColumn Error;
        private System.Windows.Forms.Button ExportStreamSet_Btn;
        private System.Windows.Forms.SaveFileDialog StreamSetsaveFileDialog;
        private System.Windows.Forms.DataGridView nonStream_GridView;
        private System.Windows.Forms.Button nonStreamBtn;
        private System.Windows.Forms.Button ExportnonStreamBtn;
        private System.Windows.Forms.DataGridViewTextBoxColumn StreamSets;
        private System.Windows.Forms.DataGridViewTextBoxColumn Durations;
    }
}