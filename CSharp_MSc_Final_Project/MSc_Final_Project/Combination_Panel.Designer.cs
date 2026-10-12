namespace MSc_Final_Project
{
    partial class Combination_Panel
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
            this.Comb_GridView = new System.Windows.Forms.DataGridView();
            this.SaveCombs_Btn = new System.Windows.Forms.Button();
            this.RemCombs_Btn = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.Comb_GridView)).BeginInit();
            this.SuspendLayout();
            // 
            // Comb_GridView
            // 
            this.Comb_GridView.AllowUserToAddRows = false;
            this.Comb_GridView.AllowUserToDeleteRows = false;
            this.Comb_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.Comb_GridView.Location = new System.Drawing.Point(12, 12);
            this.Comb_GridView.Name = "Comb_GridView";
            this.Comb_GridView.Size = new System.Drawing.Size(426, 75);
            this.Comb_GridView.TabIndex = 0;
            // 
            // SaveCombs_Btn
            // 
            this.SaveCombs_Btn.Location = new System.Drawing.Point(55, 136);
            this.SaveCombs_Btn.Name = "SaveCombs_Btn";
            this.SaveCombs_Btn.Size = new System.Drawing.Size(137, 44);
            this.SaveCombs_Btn.TabIndex = 1;
            this.SaveCombs_Btn.Text = "Save Combinations";
            this.SaveCombs_Btn.UseVisualStyleBackColor = true;
            this.SaveCombs_Btn.Click += new System.EventHandler(this.SaveCombs_Btn_Click);
            // 
            // RemCombs_Btn
            // 
            this.RemCombs_Btn.Location = new System.Drawing.Point(260, 136);
            this.RemCombs_Btn.Name = "RemCombs_Btn";
            this.RemCombs_Btn.Size = new System.Drawing.Size(143, 44);
            this.RemCombs_Btn.TabIndex = 1;
            this.RemCombs_Btn.Text = "Remove Combinations";
            this.RemCombs_Btn.UseVisualStyleBackColor = true;
            this.RemCombs_Btn.Click += new System.EventHandler(this.RemCombs_Btn_Click);
            // 
            // Combination_Panel
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(450, 211);
            this.Controls.Add(this.RemCombs_Btn);
            this.Controls.Add(this.SaveCombs_Btn);
            this.Controls.Add(this.Comb_GridView);
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(466, 250);
            this.MinimumSize = new System.Drawing.Size(466, 250);
            this.Name = "Combination_Panel";
            this.Text = "Combination_Panel";
            this.Load += new System.EventHandler(this.Combination_Panel_Load);
            ((System.ComponentModel.ISupportInitialize)(this.Comb_GridView)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.DataGridView Comb_GridView;
        private System.Windows.Forms.Button SaveCombs_Btn;
        private System.Windows.Forms.Button RemCombs_Btn;
    }
}